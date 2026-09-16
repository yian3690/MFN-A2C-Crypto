"""GitHub-style two-view Memory Fusion Network for SB3.

This extractor adapts the original three-view MFN to two market modalities:
price changes and technical indicators.  It preserves the original MFN core:
previous/current LSTM cell-state attention, a candidate-memory network, and
memory-conditioned retention/update gates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.feature_schema import INDICATOR_DIM, PRICE_DIM


def _two_layer_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Build the two-layer MLP pattern used by the original MFN."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class GitHubStyleTwoViewMFN(BaseFeaturesExtractor):
    """Encode a 20-step, two-modal market window with full MFN memory."""

    def __init__(
        self,
        observation_space,
        price_dim: int = PRICE_DIM,
        indicator_dim: int = INDICATOR_DIM,
        lstm_hidden: int = 64,
        memory_dim: int = 128,
        attention_hidden: int = 64,
        candidate_hidden: int = 64,
        gate_hidden: int = 64,
        dropout: float = 0.0,
    ):
        if len(observation_space.shape) != 2:
            raise ValueError(
                "MFN expects a 2-D observation (timesteps, features), "
                f"got {observation_space.shape}"
            )

        observation_features = observation_space.shape[1]
        if price_dim + indicator_dim != observation_features:
            raise ValueError(
                f"price_dim + indicator_dim = {price_dim + indicator_dim}, "
                f"but observation has {observation_features} features."
            )

        # The downstream A2C receives both final modality states and memory.
        fused_dim = 2 * lstm_hidden + memory_dim
        super().__init__(observation_space, features_dim=fused_dim)

        self.price_dim = price_dim
        self.indicator_dim = indicator_dim
        self.lstm_hidden = lstm_hidden
        self.memory_dim = memory_dim

        # One independent temporal encoder for each market modality.
        self.price_lstm = nn.LSTMCell(price_dim, lstm_hidden)
        self.indicator_lstm = nn.LSTMCell(indicator_dim, lstm_hidden)

        # cStar contains previous and current cell states from both modalities.
        self.cstar_dim = 4 * lstm_hidden
        self.attention_network = _two_layer_mlp(
            self.cstar_dim,
            attention_hidden,
            self.cstar_dim,
            dropout,
        )

        # Transform attended cell states into a learnable memory candidate.
        self.candidate_network = _two_layer_mlp(
            self.cstar_dim,
            candidate_hidden,
            memory_dim,
            dropout,
        )

        # Both gates condition on the attended interaction and previous memory.
        self.gate_input_dim = self.cstar_dim + memory_dim
        self.retention_gate = _two_layer_mlp(
            self.gate_input_dim,
            gate_hidden,
            memory_dim,
            dropout,
        )
        self.update_gate = _two_layer_mlp(
            self.gate_input_dim,
            gate_hidden,
            memory_dim,
            dropout,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return fused final hidden states and shared MFN memory."""
        x = observations.float()
        batch_size, timesteps, _ = x.shape
        device = x.device
        dtype = x.dtype

        x_price = x[:, :, : self.price_dim]
        x_indicator = x[:, :, self.price_dim :]

        h_price = torch.zeros(
            batch_size, self.lstm_hidden, device=device, dtype=dtype
        )
        c_price = torch.zeros_like(h_price)
        h_indicator = torch.zeros_like(h_price)
        c_indicator = torch.zeros_like(h_price)
        memory = torch.zeros(
            batch_size, self.memory_dim, device=device, dtype=dtype
        )

        for timestep in range(timesteps):
            previous_c_price = c_price
            previous_c_indicator = c_indicator

            h_price, c_price = self.price_lstm(
                x_price[:, timestep, :],
                (h_price, c_price),
            )
            h_indicator, c_indicator = self.indicator_lstm(
                x_indicator[:, timestep, :],
                (h_indicator, c_indicator),
            )

            c_star = torch.cat(
                [
                    previous_c_price,
                    previous_c_indicator,
                    c_price,
                    c_indicator,
                ],
                dim=1,
            )
            attention = F.softmax(self.attention_network(c_star), dim=1)
            attended = attention * c_star

            candidate = torch.tanh(self.candidate_network(attended))
            gate_input = torch.cat([attended, memory], dim=1)
            retention = torch.sigmoid(self.retention_gate(gate_input))
            update = torch.sigmoid(self.update_gate(gate_input))
            memory = retention * memory + update * candidate

        return torch.cat([h_price, h_indicator, memory], dim=1)
