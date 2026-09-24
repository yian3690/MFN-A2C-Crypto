"""4H實驗用的資產感知MFN與無MGM Self-Attention特徵擷取器。"""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.feature_schema import INDICATOR_DIM, PRICE_DIM
from src.mfn_github_extractor import GitHubStyleTwoViewMFN


def _validate_asset_layout(
    observation_space,
    price_dim: int,
    indicator_dim: int,
    indicators_per_asset: int,
) -> int:
    """驗證輸入能依資產切成價格與相同數量的技術指標。"""
    if len(observation_space.shape) != 2:
        raise ValueError(
            "Extractor expects (timesteps, features), "
            f"got {observation_space.shape}."
        )
    if price_dim + indicator_dim != observation_space.shape[1]:
        raise ValueError("price_dim + indicator_dim與觀察特徵數不一致。")
    if indicator_dim % indicators_per_asset != 0:
        raise ValueError("indicator_dim無法依每項資產指標數整除。")
    asset_count = indicator_dim // indicators_per_asset
    if asset_count != price_dim:
        raise ValueError("每項資產必須各有一個價格特徵與相同數量技術指標。")
    return asset_count


class AssetWiseAttentionMFN(GitHubStyleTwoViewMFN):
    """先做資產間Attention，再送進原雙LSTM／DMAN／MGM。"""

    def __init__(
        self,
        observation_space,
        price_dim: int = PRICE_DIM,
        indicator_dim: int = INDICATOR_DIM,
        indicators_per_asset: int = 5,
        asset_embedding_dim: int = 16,
        asset_attention_heads: int = 2,
        asset_attention_dropout: float = 0.0,
        **mfn_kwargs,
    ):
        asset_count = _validate_asset_layout(
            observation_space,
            price_dim,
            indicator_dim,
            indicators_per_asset,
        )
        if asset_embedding_dim % asset_attention_heads != 0:
            raise ValueError("asset_embedding_dim必須可被attention heads整除。")
        super().__init__(
            observation_space,
            price_dim=price_dim,
            indicator_dim=indicator_dim,
            **mfn_kwargs,
        )
        self.asset_count = asset_count
        self.indicators_per_asset = indicators_per_asset
        self.asset_input = nn.Linear(indicators_per_asset, asset_embedding_dim)
        self.asset_identity = nn.Parameter(
            torch.zeros(1, 1, asset_count, asset_embedding_dim)
        )
        nn.init.normal_(self.asset_identity, mean=0.0, std=0.02)
        self.asset_attention = nn.MultiheadAttention(
            asset_embedding_dim,
            asset_attention_heads,
            dropout=asset_attention_dropout,
            batch_first=True,
        )
        self.asset_norm = nn.LayerNorm(asset_embedding_dim)
        self.asset_output = nn.Linear(asset_embedding_dim, indicators_per_asset)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()
        batch_size, timesteps, _ = x.shape
        price = x[:, :, : self.price_dim]
        indicator = x[:, :, self.price_dim :].reshape(
            batch_size,
            timesteps,
            self.asset_count,
            self.indicators_per_asset,
        )

        tokens = self.asset_input(indicator) + self.asset_identity
        flat_tokens = tokens.reshape(
            batch_size * timesteps,
            self.asset_count,
            -1,
        )
        attended, _ = self.asset_attention(
            flat_tokens,
            flat_tokens,
            flat_tokens,
            need_weights=False,
        )
        attended = self.asset_norm(flat_tokens + attended)
        # Residual保留原技術指標，同時加入跨資產關係。
        indicator = indicator + self.asset_output(attended).reshape_as(indicator)
        fused_input = torch.cat(
            [price, indicator.reshape(batch_size, timesteps, -1)],
            dim=-1,
        )
        return super().forward(fused_input)


class AssetTemporalSelfAttentionA2C(BaseFeaturesExtractor):
    """共享資產LSTM＋Asset-wise Self-Attention；完全不使用DMAN/MGM。"""

    def __init__(
        self,
        observation_space,
        price_dim: int = PRICE_DIM,
        indicator_dim: int = INDICATOR_DIM,
        indicators_per_asset: int = 5,
        temporal_hidden: int = 32,
        attention_heads: int = 2,
        output_dim: int = 128,
        dropout: float = 0.0,
    ):
        asset_count = _validate_asset_layout(
            observation_space,
            price_dim,
            indicator_dim,
            indicators_per_asset,
        )
        if temporal_hidden % attention_heads != 0:
            raise ValueError("temporal_hidden必須可被attention heads整除。")
        super().__init__(observation_space, features_dim=output_dim)
        self.price_dim = price_dim
        self.asset_count = asset_count
        self.indicators_per_asset = indicators_per_asset
        self.temporal_encoder = nn.LSTM(
            input_size=1 + indicators_per_asset,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.asset_identity = nn.Parameter(
            torch.zeros(1, asset_count, temporal_hidden)
        )
        nn.init.normal_(self.asset_identity, mean=0.0, std=0.02)
        self.asset_attention = nn.MultiheadAttention(
            temporal_hidden,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.asset_norm = nn.LayerNorm(temporal_hidden)
        self.output = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(asset_count * temporal_hidden, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()
        batch_size, timesteps, _ = x.shape
        price = x[:, :, : self.price_dim].permute(0, 2, 1).unsqueeze(-1)
        indicator = x[:, :, self.price_dim :].reshape(
            batch_size,
            timesteps,
            self.asset_count,
            self.indicators_per_asset,
        ).permute(0, 2, 1, 3)
        asset_sequences = torch.cat([price, indicator], dim=-1).reshape(
            batch_size * self.asset_count,
            timesteps,
            1 + self.indicators_per_asset,
        )
        _, (hidden, _) = self.temporal_encoder(asset_sequences)
        tokens = hidden[-1].reshape(batch_size, self.asset_count, -1)
        tokens = tokens + self.asset_identity
        attended, _ = self.asset_attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )
        tokens = self.asset_norm(tokens + attended)
        return self.output(tokens)
