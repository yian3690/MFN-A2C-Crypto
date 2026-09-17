"""
Two-view MFN feature extractor for Stable-Baselines3.

Thesis-equation design:
    Price-change modality  -> LSTMCell
    Technical-indicator modality -> LSTMCell
    Hidden-state differences -> single-linear DMAN attention
    Attended changes -> single-linear MGM gates -> shared memory
    Fused representation -> SB3 A2C policy/value networks

The original MFN code uploaded with the project was a 3-view implementation
with a third modality commented/partially removed. This file implements the
clean 2-view version directly rather than keeping the half-removed branch.

Input shape expected by SB3:
    (batch, 20, 30)
where:
    first 5 columns   = 5 asset price-relative features
    last 25 columns   = 5 assets x (4 technical indicators + relative strength)

If your CSV columns are arranged differently, adjust split_dims below.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.feature_schema import INDICATOR_DIM, PRICE_DIM


class TwoViewMFN(BaseFeaturesExtractor):
    """Convert a two-modal market window into features for A2C.

    This class implements the paper's two-view MFN.  It does not choose
    portfolio weights itself: SB3 passes its output to the A2C actor and
    critic, which choose and evaluate the portfolio action.
    """
    def __init__(
        self,
        observation_space,
        price_dim: int = PRICE_DIM,
        indicator_dim: int = INDICATOR_DIM,
        lstm_hidden: int = 64,
        memory_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.0,
    ):
        """建立兩個 LSTM、跨模態注意力、共享記憶閘門及輸出投影層。"""
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

        # Each modality has its own temporal state.  The paper adapts the
        # original three-view MFN to price changes and technical indicators.
        self.price_lstm = nn.LSTMCell(price_dim, lstm_hidden)
        self.indicator_lstm = nn.LSTMCell(indicator_dim, lstm_hidden)

        # Thesis equations 4.5-4.8 concatenate the two modalities' hidden-
        # state differences: [h_price(t)-h_price(t-1),
        # h_indicator(t)-h_indicator(t-1)].
        delta_dim = 2 * lstm_hidden

        # Equation 4.10 writes tanh(attended_delta) directly into memory, so
        # the attended vector and shared memory must have the same width.
        if memory_dim != delta_dim:
            raise ValueError(
                "Thesis MGM requires memory_dim == 2 * lstm_hidden; "
                f"got memory_dim={memory_dim} and 2*lstm_hidden={delta_dim}."
            )

        # Equation 4.7: alpha_t = softmax(W_a * delta_h_t + b_a).
        self.attention = nn.Linear(delta_dim, delta_dim)

        # Equation 4.9: both gates depend only on the current attended
        # hidden-state difference c_t.
        self.retention_gate = nn.Linear(delta_dim, memory_dim)
        self.update_gate = nn.Linear(delta_dim, memory_dim)

        # Final representation sent to SB3 A2C.
        final_dim = 2 * lstm_hidden + memory_dim
        self.out1 = nn.Linear(final_dim, output_dim)
        self.out_dropout = nn.Dropout(dropout)

        self._features_dim = output_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 provides (batch, 20 historical steps, 25 features).
        """將一批歷史市場觀察值編碼成供 A2C 使用的特徵向量。"""
        x = observations.float()

        batch_size, timesteps, _ = x.shape
        device = x.device

        # Feature preparation fixes this order: 5 price-relative columns,
        # followed by 25 indicator/relative-strength columns.
        x_price = x[:, :, : self.price_dim]
        x_indicator = x[:, :, self.price_dim :]

        h_price = torch.zeros(batch_size, self.hidden, device=device)
        c_price = torch.zeros(batch_size, self.hidden, device=device)

        h_indicator = torch.zeros(batch_size, self.hidden, device=device)
        c_indicator = torch.zeros(batch_size, self.hidden, device=device)

        memory = torch.zeros(batch_size, self.memory_dim, device=device)

        # Process the history chronologically and update both modality
        # memories plus the shared MFN memory at each time step.
        for t in range(timesteps):
            prev_h_price = h_price
            prev_h_indicator = h_indicator

            h_price, c_price = self.price_lstm(
                x_price[:, t, :], (h_price, c_price)
            )
            h_indicator, c_indicator = self.indicator_lstm(
                x_indicator[:, t, :], (h_indicator, c_indicator)
            )

            # Equation 4.5-4.6: delta_h_t contains the adjacent hidden-state
            # differences from both modalities.
            delta_h = torch.cat(
                [
                    h_price - prev_h_price,
                    h_indicator - prev_h_indicator,
                ],
                dim=1,
            )

            attention_logits = self.attention(delta_h)
            attention = F.softmax(attention_logits, dim=1)
            attended = attention * delta_h

            gamma1 = torch.sigmoid(self.retention_gate(attended))
            gamma2 = torch.sigmoid(self.update_gate(attended))

            # Equation 4.10:
            # u_t = gamma1 * u_(t-1) + gamma2 * tanh(c_t).
            memory = gamma1 * memory + gamma2 * torch.tanh(attended)

        # The actor/critic receive both final LSTM states and shared memory.
        fused = torch.cat([h_price, h_indicator, memory], dim=1)
        features = self.out1(self.out_dropout(F.relu(fused)))

        return features
