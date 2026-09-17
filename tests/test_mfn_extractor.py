"""Tests for the thesis-equation two-view MFN extractor."""

import unittest

import gymnasium as gym
import torch

from src.mfn_sb3_extractor import TwoViewMFN


class TwoViewMFNTests(unittest.TestCase):
    def setUp(self):
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(20, 30),
            dtype=float,
        )

    def test_thesis_dman_and_mgm_dimensions(self):
        model = TwoViewMFN(
            self.observation_space,
            price_dim=5,
            indicator_dim=25,
            lstm_hidden=8,
            memory_dim=16,
            output_dim=12,
        )

        # Two 8-D hidden-state differences are concatenated into delta_h.
        self.assertEqual(model.attention.in_features, 16)
        self.assertEqual(model.attention.out_features, 16)

        # Thesis equation 4.9 gates receive only the attended delta.
        self.assertEqual(model.retention_gate.in_features, 16)
        self.assertEqual(model.update_gate.in_features, 16)

    def test_forward_shape_and_gradients(self):
        model = TwoViewMFN(
            self.observation_space,
            price_dim=5,
            indicator_dim=25,
            lstm_hidden=8,
            memory_dim=16,
            output_dim=12,
        )
        observations = torch.randn(4, 20, 30, requires_grad=True)

        output = model(observations)
        self.assertEqual(tuple(output.shape), (4, 12))
        self.assertTrue(torch.isfinite(output).all())

        output.square().mean().backward()
        for parameter_name in (
            "attention.weight",
            "retention_gate.weight",
            "update_gate.weight",
        ):
            parameter = dict(model.named_parameters())[parameter_name]
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_memory_dimension_must_match_delta_dimension(self):
        with self.assertRaisesRegex(
            ValueError,
            "memory_dim == 2 \\* lstm_hidden",
        ):
            TwoViewMFN(
                self.observation_space,
                lstm_hidden=8,
                memory_dim=12,
            )


if __name__ == "__main__":
    unittest.main()
