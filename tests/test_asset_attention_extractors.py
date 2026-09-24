"""4H Asset-wise Attention 特徵擷取器的基本結構測試。"""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch

from src.asset_attention_extractors import (
    AssetTemporalSelfAttentionA2C,
    AssetWiseAttentionMFN,
)


class AssetAttentionExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20, 30),
            dtype=np.float32,
        )
        self.observations = torch.randn(3, 20, 30, requires_grad=True)

    def test_asset_attention_mfn_forward_and_gradient(self) -> None:
        extractor = AssetWiseAttentionMFN(
            self.observation_space,
            price_dim=5,
            indicator_dim=25,
            indicators_per_asset=5,
            asset_embedding_dim=16,
            asset_attention_heads=2,
            lstm_hidden=64,
            memory_dim=128,
        )
        output = extractor(self.observations)
        self.assertEqual(output.shape, (3, 256))
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(extractor.asset_input.weight.grad)

    def test_no_mgm_self_attention_forward_and_gradient(self) -> None:
        extractor = AssetTemporalSelfAttentionA2C(
            self.observation_space,
            price_dim=5,
            indicator_dim=25,
            indicators_per_asset=5,
            temporal_hidden=32,
            attention_heads=2,
            output_dim=128,
        )
        output = extractor(self.observations)
        self.assertEqual(output.shape, (3, 128))
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(extractor.temporal_encoder.weight_ih_l0.grad)

    def test_rejects_inconsistent_asset_layout(self) -> None:
        with self.assertRaises(ValueError):
            AssetTemporalSelfAttentionA2C(
                self.observation_space,
                price_dim=5,
                indicator_dim=25,
                indicators_per_asset=4,
            )


if __name__ == "__main__":
    unittest.main()
