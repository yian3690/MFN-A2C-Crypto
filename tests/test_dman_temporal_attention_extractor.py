"""雙LSTM＋DMAN＋Temporal Self-Attention extractor測試。"""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch

from src.dman_temporal_attention_extractor import (
    DualLSTMDMANTemporalAttention,
)


class DMANTemporalAttentionExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20, 30),
            dtype=np.float32,
        )

    def test_forward_shape_finite_and_gradients(self) -> None:
        extractor = DualLSTMDMANTemporalAttention(
            self.space,
            price_dim=5,
            indicator_dim=25,
            lstm_hidden=64,
            dman_hidden=64,
            temporal_dim=128,
            temporal_attention_heads=4,
        )
        observations = torch.randn(3, 20, 30, requires_grad=True)
        output = extractor(observations)
        self.assertEqual(output.shape, (3, 256))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(extractor.price_lstm.weight_ih.grad)
        self.assertIsNotNone(extractor.dman_attention_network[0].weight.grad)
        self.assertIsNotNone(extractor.temporal_attention.in_proj_weight.grad)
        self.assertIsNotNone(extractor.pooling_score.weight.grad)

    def test_contains_no_mgm_components(self) -> None:
        extractor = DualLSTMDMANTemporalAttention(self.space)
        names = set(dict(extractor.named_modules()))
        self.assertNotIn("candidate_network", names)
        self.assertNotIn("retention_gate", names)
        self.assertNotIn("update_gate", names)

    def test_rejects_invalid_attention_heads(self) -> None:
        with self.assertRaises(ValueError):
            DualLSTMDMANTemporalAttention(
                self.space,
                temporal_dim=127,
                temporal_attention_heads=4,
            )


if __name__ == "__main__":
    unittest.main()
