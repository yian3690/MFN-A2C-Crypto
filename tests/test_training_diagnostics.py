"""Tests for rollout-diagnostics interpretation."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.training_diagnostics import diagnose_training_file


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_flags_weak_critic_and_static_equal_weight_policy(self):
        frame = pd.DataFrame(
            {
                "timesteps": [540, 1080],
                "reward_std": [1e-6, 1e-6],
                "reward_abs_p99": [1e-5, 1e-5],
                "explained_variance": [-0.2, -0.1],
                "mfn_gradient_norm": [1e-10, 1e-10],
                "allocation_entropy_mean": [0.999, 0.999],
                "policy_equal_weight_l1_mean": [0.01, 0.01],
                "policy_weight_temporal_std_mean": [0.001, 0.001],
                "turnover_mean": [0.02, 0.02],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mfn_a2c_diagnostics.csv"
            frame.to_csv(path, index=False)
            summary, warnings = diagnose_training_file(path)

        combined = " ".join(warnings)
        self.assertEqual(summary["rollouts"], 2.0)
        self.assertIn("reward variation", combined)
        self.assertIn("explained variance", combined)
        self.assertIn("gradients", combined)
        self.assertIn("equal weight", combined)
        self.assertIn("barely changes", combined)

    def test_baseline_does_not_report_mfn_gradient_failure(self):
        frame = pd.DataFrame(
            {
                "timesteps": [540],
                "reward_std": [1.0],
                "reward_abs_p99": [2.0],
                "explained_variance": [0.5],
                "mfn_gradient_norm": [0.0],
                "allocation_entropy_mean": [0.8],
                "policy_equal_weight_l1_mean": [0.2],
                "policy_weight_temporal_std_mean": [0.02],
                "turnover_mean": [0.02],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a2c_baseline_diagnostics.csv"
            frame.to_csv(path, index=False)
            summary, warnings = diagnose_training_file(path)

        self.assertTrue(pd.isna(summary["mfn_gradient_norm_recent"]))
        self.assertNotIn("gradients", " ".join(warnings))


if __name__ == "__main__":
    unittest.main()
