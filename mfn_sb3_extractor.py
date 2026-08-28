"""
Two-view MFN feature extractor for Stable-Baselines3.

Paper-aligned design:
    Price-change modality  -> LSTMCell
    Technical-indicator modality -> LSTMCell
    Two-view memory fusion -> attention (DMAN-like) + gated memory (MGM-like)
    Fused representation -> SB3 A2C policy/value networks

The original MFN code uploaded with the project was a 3-view implementation
with a third modality commented/partially removed. This file implements the
clean 2-view version directly rather than keeping the half-removed branch.

Input shape expected by SB3:
    (batch, 20, 32)
where:
    first 16 columns  = 4 assets x 4 price-change features
    last 16 columns   = 4 assets x 4 technical indicators

If your CSV columns are arranged differently, adjust split_dims below.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TwoViewMFN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        price_dim: int = 16,
        indicator_dim: int = 16,
        lstm_hidden: int = 64,
        memory_dim: int = 128,
        att_hidden: int = 64,
        gate_hidden: int = 64,
        output_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim=output_dim)

        if len(observation_space.shape) != 2:
            raise ValueError(
                f"MFN expects a 2-D observation (timesteps, features), "
                f"got {observation_space.shape}"
            )

        obs_features = observation_space.shape[1]
        if price_dim + indicator_dim != obs_features:
            raise ValueError(
                f"price_dim + indicator_dim = {price_dim + indicator_dim}, "
                f"but observation has {obs_features} features."
            )

        self.price_dim = price_dim
        self.indicator_dim = indicator_dim
        self.hidden = lstm_hidden
        self.memory_dim = memory_dim

        # Two modalities only: the third original MFN modality is removed.
        self.price_lstm = nn.LSTMCell(price_dim, lstm_hidden)
        self.indicator_lstm = nn.LSTMCell(indicator_dim, lstm_hidden)

        # cStar = previous two-view cell states + current two-view cell states
        cstar_dim = 4 * lstm_hidden

        # Attention block: produces an attention vector over cStar.
        self.att1 = nn.Linear(cstar_dim, att_hidden)
        self.att2 = nn.Linear(att_hidden, cstar_dim)

        # Attended cStar -> shared candidate memory.
        self.att_out1 = nn.Linear(cstar_dim, att_hidden)
        self.att_out2 = nn.Linear(att_hidden, memory_dim)

        # MGM-style gates: previous memory + attended representation.
        gate_in_dim = cstar_dim + memory_dim
        self.gamma1_1 = nn.Linear(gate_in_dim, gate_hidden)
        self.gamma1_2 = nn.Linear(gate_hidden, memory_dim)

        self.gamma2_1 = nn.Linear(gate_in_dim, gate_hidden)
        self.gamma2_2 = nn.Linear(gate_hidden, memory_dim)

        # Final representation sent to SB3 A2C.
        final_dim = 2 * lstm_hidden + memory_dim
        self.out1 = nn.Linear(final_dim, output_dim)
        self.out_dropout = nn.Dropout(dropout)

        self._features_dim = output_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 may provide float32 tensors directly.
        x = observations.float()

        batch_size, timesteps, _ = x.shape
        device = x.device

        x_price = x[:, :, : self.price_dim]
        x_indicator = x[:, :, self.price_dim :]

        h_price = torch.zeros(batch_size, self.hidden, device=device)
        c_price = torch.zeros(batch_size, self.hidden, device=device)

        h_indicator = torch.zeros(batch_size, self.hidden, device=device)
        c_indicator = torch.zeros(batch_size, self.hidden, device=device)

        memory = torch.zeros(batch_size, self.memory_dim, device=device)

        for t in range(timesteps):
            prev_c = torch.cat([c_price, c_indicator], dim=1)

            h_price, c_price = self.price_lstm(
                x_price[:, t, :], (h_price, c_price)
            )
            h_indicator, c_indicator = self.indicator_lstm(
                x_indicator[:, t, :], (h_indicator, c_indicator)
            )

            new_c = torch.cat([c_price, c_indicator], dim=1)
            c_star = torch.cat([prev_c, new_c], dim=1)

            attention_logits = self.att2(
                F.relu(self.att1(c_star))
            )
            attention = F.softmax(attention_logits, dim=1)
            attended = attention * c_star

            candidate = torch.tanh(
                self.att_out2(F.relu(self.att_out1(attended)))
            )

            both = torch.cat([attended, memory], dim=1)

            gamma1 = torch.sigmoid(
                self.gamma1_2(F.relu(self.gamma1_1(both)))
            )
            gamma2 = torch.sigmoid(
                self.gamma2_2(F.relu(self.gamma2_1(both)))
            )

            memory = gamma1 * memory + gamma2 * candidate

        fused = torch.cat([h_price, h_indicator, memory], dim=1)
        features = self.out1(self.out_dropout(F.relu(fused)))

        return features
