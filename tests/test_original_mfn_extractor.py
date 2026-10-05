"""原始雙模態 MFN extractor 的結構與梯度測試。"""

import unittest

import gymnasium as gym
import torch

from src.original_mfn_extractor import OriginalTwoViewMFN


class OriginalTwoViewMFNTests(unittest.TestCase):
    def setUp(self):
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(20, 30),
            dtype=float,
        )

    def test_dual_lstm_dman_and_mgm_dimensions(self):
        model = OriginalTwoViewMFN(
            self.observation_space,
            price_dim=5,
            indicator_dim=25,
            lstm_hidden=8,
            memory_dim=12,
            attention_hidden=10,
            candidate_hidden=9,
            gate_hidden=7,
        )
        self.assertEqual(model.cstar_dim, 32)
        self.assertEqual(model.dman_attention_network[0].in_features, 32)
        self.assertEqual(model.candidate_network[-1].out_features, 12)
        self.assertEqual(model.gate_input_dim, 44)
        self.assertEqual(model.retention_gate[0].in_features, 44)
        self.assertEqual(model.update_gate[-1].out_features, 12)
        self.assertEqual(model.features_dim, 28)

    def test_forward_shape_finiteness_and_gradients(self):
        model = OriginalTwoViewMFN(
            self.observation_space,
            price_dim=5,
            indicator_dim=25,
            lstm_hidden=8,
            memory_dim=12,
            attention_hidden=10,
            candidate_hidden=9,
            gate_hidden=7,
        )
        observations = torch.randn(4, 20, 30, requires_grad=True)
        output = model(observations)
        self.assertEqual(tuple(output.shape), (4, 28))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        parameters = dict(model.named_parameters())
        for name in (
            "dman_attention_network.0.weight",
            "candidate_network.0.weight",
            "retention_gate.0.weight",
            "update_gate.0.weight",
        ):
            self.assertIsNotNone(parameters[name].grad)
            self.assertTrue(torch.isfinite(parameters[name].grad).all())

    def test_modal_dimensions_must_match_observation(self):
        with self.assertRaisesRegex(ValueError, "observation has 30 features"):
            OriginalTwoViewMFN(
                self.observation_space,
                price_dim=4,
                indicator_dim=25,
            )


if __name__ == "__main__":
    unittest.main()
