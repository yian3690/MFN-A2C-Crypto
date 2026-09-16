"""Tests for direct Dirichlet portfolio-weight actions."""

import unittest

import numpy as np
import torch
from gymnasium import spaces

from src.simplex_policy import (
    DirichletDistribution,
    SimplexActorCriticPolicy,
)


class DirichletDistributionTests(unittest.TestCase):
    def test_sample_is_a_valid_simplex(self):
        distribution = DirichletDistribution(action_dim=5)
        raw_concentration = torch.randn(32, 5)
        distribution.proba_distribution(raw_concentration)

        actions = distribution.sample()

        self.assertEqual(tuple(actions.shape), (32, 5))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue((actions > 0.0).all())
        self.assertTrue(
            torch.allclose(
                actions.sum(dim=1),
                torch.ones(32),
                atol=1e-6,
            )
        )

    def test_deterministic_action_is_dirichlet_mean(self):
        distribution = DirichletDistribution(action_dim=5)
        raw_concentration = torch.tensor(
            [[-2.0, -1.0, 0.0, 1.0, 2.0]]
        )
        distribution.proba_distribution(raw_concentration)

        action = distribution.mode()

        self.assertTrue((action > 0.0).all())
        self.assertTrue(
            torch.allclose(action.sum(dim=1), torch.ones(1), atol=1e-7)
        )
        np.testing.assert_allclose(
            action.detach().numpy(),
            (
                distribution.concentration
                / distribution.concentration.sum(dim=1, keepdim=True)
            ).detach().numpy(),
        )

    def test_log_prob_and_entropy_are_finite(self):
        distribution = DirichletDistribution(action_dim=5)
        distribution.proba_distribution(torch.zeros(4, 5))
        actions = distribution.sample()

        self.assertTrue(torch.isfinite(distribution.log_prob(actions)).all())
        self.assertTrue(torch.isfinite(distribution.entropy()).all())

    def test_policy_has_no_misleading_gaussian_std(self):
        policy = SimplexActorCriticPolicy(
            observation_space=spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            ),
            action_space=spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5,),
                dtype=np.float32,
            ),
            lr_schedule=lambda _: 7e-4,
        )

        self.assertFalse(hasattr(policy, "log_std"))
        action, _, log_prob = policy(torch.zeros(3, 4))
        self.assertEqual(tuple(action.shape), (3, 5))
        self.assertTrue(torch.allclose(action.sum(dim=1), torch.ones(3)))
        self.assertTrue(torch.isfinite(log_prob).all())


if __name__ == "__main__":
    unittest.main()
