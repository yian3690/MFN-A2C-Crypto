"""Tests for the GitHub-style two-view MFN extractor."""

import unittest

import gymnasium as gym
import torch

from src.mfn_github_extractor import GitHubStyleTwoViewMFN


class GitHubStyleTwoViewMFNTests(unittest.TestCase):
    def setUp(self):
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(20, 25),
            dtype=float,
        )

    def test_original_mfn_dimensions_are_preserved_for_two_views(self):
        model = GitHubStyleTwoViewMFN(
            self.observation_space,
            lstm_hidden=8,
            memory_dim=12,
            attention_hidden=10,
            candidate_hidden=9,
            gate_hidden=7,
        )

        # Two modalities x previous/current cell states x 8 units.
        self.assertEqual(model.cstar_dim, 32)
        self.assertEqual(model.attention_network[0].in_features, 32)
        self.assertEqual(model.attention_network[-1].out_features, 32)

        # Candidate memory can have a width independent of cStar.
        self.assertEqual(model.candidate_network[-1].out_features, 12)

        # Each gate reads attended cStar together with previous memory.
        self.assertEqual(model.gate_input_dim, 44)
        self.assertEqual(model.retention_gate[0].in_features, 44)
        self.assertEqual(model.update_gate[-1].out_features, 12)

        # Final hidden states (8 + 8) and shared memory (12) go to A2C.
        self.assertEqual(model.features_dim, 28)

    def test_forward_shape_finiteness_and_gradients(self):
        model = GitHubStyleTwoViewMFN(
            self.observation_space,
            lstm_hidden=8,
            memory_dim=12,
            attention_hidden=10,
            candidate_hidden=9,
            gate_hidden=7,
        )
        observations = torch.randn(4, 20, 25, requires_grad=True)

        output = model(observations)
        self.assertEqual(tuple(output.shape), (4, 28))
        self.assertTrue(torch.isfinite(output).all())

        output.square().mean().backward()
        parameters = dict(model.named_parameters())
        for parameter_name in (
            "attention_network.0.weight",
            "candidate_network.0.weight",
            "retention_gate.0.weight",
            "update_gate.0.weight",
        ):
            gradient = parameters[parameter_name].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_modal_feature_dimensions_must_match_observation(self):
        with self.assertRaisesRegex(ValueError, "observation has 25 features"):
            GitHubStyleTwoViewMFN(
                self.observation_space,
                price_dim=4,
                indicator_dim=20,
            )


if __name__ == "__main__":
    unittest.main()
