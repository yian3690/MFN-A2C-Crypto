"""保留雙LSTM與DMAN、以Temporal Self-Attention取代MGM。"""

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
    """建立與原MFN之DMAN相同形式的兩層MLP。"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class DualLSTMDMANTemporalAttention(BaseFeaturesExtractor):
    """雙模態LSTM＋DMAN＋時間Self-Attention，不使用MGM。

    保留原MFN的兩個LSTMCell與DMAN：每個時間點以價格、技術指標的
    前一期／當期cell state形成cStar，再由DMAN產生融合向量。不同之處是
    不建立candidate memory、retention gate或update gate；完整DMAN序列改由
    Temporal Self-Attention建模，最後以可學習attention pooling彙整。
    """

    def __init__(
        self,
        observation_space,
        price_dim: int = PRICE_DIM,
        indicator_dim: int = INDICATOR_DIM,
        lstm_hidden: int = 64,
        dman_hidden: int = 64,
        temporal_dim: int = 128,
        temporal_attention_heads: int = 4,
        dropout: float = 0.0,
    ):
        if len(observation_space.shape) != 2:
            raise ValueError(
                "Extractor expects (timesteps, features), "
                f"got {observation_space.shape}."
            )
        if price_dim + indicator_dim != observation_space.shape[1]:
            raise ValueError("price_dim + indicator_dim與觀察特徵數不一致。")
        if temporal_dim % temporal_attention_heads != 0:
            raise ValueError("temporal_dim必須可被attention heads整除。")

        # 輸出維度維持與原MFN一致：兩個64維hidden＋128維融合表示=256。
        features_dim = 2 * lstm_hidden + temporal_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.price_dim = price_dim
        self.indicator_dim = indicator_dim
        self.lstm_hidden = lstm_hidden
        self.max_timesteps = observation_space.shape[0]

        # 完整保留原MFN的價格模態與技術指標模態LSTM。
        self.price_lstm = nn.LSTMCell(price_dim, lstm_hidden)
        self.indicator_lstm = nn.LSTMCell(indicator_dim, lstm_hidden)

        # 完整保留DMAN：注意前一期與當期、兩種模態的cell state。
        self.cstar_dim = 4 * lstm_hidden
        self.dman_attention_network = _two_layer_mlp(
            self.cstar_dim,
            dman_hidden,
            self.cstar_dim,
            dropout,
        )

        # 將DMAN的256維融合向量投影成較精簡的時間token。
        self.temporal_projection = nn.Linear(self.cstar_dim, temporal_dim)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.max_timesteps, temporal_dim)
        )
        nn.init.normal_(self.temporal_position, mean=0.0, std=0.02)
        self.temporal_attention = nn.MultiheadAttention(
            temporal_dim,
            temporal_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(temporal_dim)

        # Pooling分數由資料學習，而非固定只取最後一個時間點或簡單平均。
        self.pooling_score = nn.Linear(temporal_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()
        batch_size, timesteps, _ = x.shape
        if timesteps > self.max_timesteps:
            raise ValueError(
                f"輸入時間長度{timesteps}超過初始化長度{self.max_timesteps}。"
            )

        x_price = x[:, :, : self.price_dim]
        x_indicator = x[:, :, self.price_dim :]
        device, dtype = x.device, x.dtype

        h_price = torch.zeros(
            batch_size, self.lstm_hidden, device=device, dtype=dtype
        )
        c_price = torch.zeros_like(h_price)
        h_indicator = torch.zeros_like(h_price)
        c_indicator = torch.zeros_like(h_price)
        attended_sequence: list[torch.Tensor] = []

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
            dman_weights = F.softmax(
                self.dman_attention_network(c_star),
                dim=1,
            )
            attended_sequence.append(dman_weights * c_star)

        # [batch, time, cStar] → Temporal Self-Attention。
        dman_sequence = torch.stack(attended_sequence, dim=1)
        tokens = self.temporal_projection(dman_sequence)
        tokens = tokens + self.temporal_position[:, :timesteps, :]
        attended, _ = self.temporal_attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )
        tokens = self.temporal_norm(tokens + attended)

        # 對時間token做可學習加權池化。
        pooling_weights = F.softmax(self.pooling_score(tokens), dim=1)
        temporal_summary = torch.sum(pooling_weights * tokens, dim=1)

        # 保留兩模態最後hidden state，並以時間注意力摘要取代MGM memory。
        return torch.cat([h_price, h_indicator, temporal_summary], dim=1)
