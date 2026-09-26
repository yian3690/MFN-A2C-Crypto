"""4H DQN 標籤、續訓與比較步數回歸測試。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config_4h as cfg  # noqa: E402
from compare_experiment2 import load_curve, result_files  # noqa: E402
from train_dqn import (  # noqa: E402
    checkpoint_prefix, checkpoint_timesteps, find_latest_checkpoint,
    normalized_model_name, parse_args, replay_buffer_path,
)


class DQNFourHourWorkflowTests(unittest.TestCase):
    def test_dqn_tag_contains_specific_hyperparameters(self):
        for token in ("ws", "cap", "buf", "start", "tf", "tgt", "ex", "lr"):
            self.assertIn(token, cfg.DQN_RUN_TAG)
        self.assertNotIn("gaussian", cfg.DQN_RUN_TAG)
        self.assertNotIn("normadv", cfg.DQN_RUN_TAG)
        self.assertEqual(cfg.DQN_LEARNING_RATE, 7e-4)
        self.assertEqual(cfg.DQN_TRAIN_FREQUENCY, 4)
        self.assertEqual(cfg.DQN_GRADIENT_STEPS, 1)
        self.assertEqual(cfg.DQN_EXPLORATION_FRACTION, 0.3)
        self.assertEqual(
            cfg.dqn_algorithm_kwargs()["learning_rate"],
            cfg.DQN_LEARNING_RATE,
        )

    def test_resume_checkpoint_is_independent_of_target_steps(self):
        self.assertNotIn(f"_{cfg.STEP_TAG.upper()}_", normalized_model_name())
        self.assertTrue(checkpoint_prefix().endswith("_RESUME"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / f"{checkpoint_prefix()}_300000_steps.zip"
            later = root / f"{checkpoint_prefix()}_600000_steps.zip"
            first.touch()
            later.touch()
            self.assertEqual(checkpoint_timesteps(first), 300_000)
            self.assertEqual(find_latest_checkpoint(root, 500_000), first)
            self.assertEqual(find_latest_checkpoint(root, 900_000), later)
            self.assertEqual(
                replay_buffer_path(later).name,
                f"{checkpoint_prefix()}_replay_buffer_600000_steps.pkl",
            )

    def test_resume_must_be_explicit(self):
        self.assertFalse(parse_args([]).resume)
        self.assertTrue(parse_args(["--resume"]).resume)

    def test_experiment2_can_select_another_step_budget(self):
        files = result_files(300_000)
        self.assertTrue(all(
            "_300k_" in name
            for label, name in files.items()
            if label != "Buy and Hold"
        ))
    def test_experiment2_recomputes_three_plot_series_from_pv(self):
        with tempfile.TemporaryDirectory() as directory:
            original = cfg.MODEL_RESULTS
            cfg.MODEL_RESULTS = Path(directory)
            try:
                path = cfg.MODEL_RESULTS / "curve.csv"
                path.write_text(
                    "timestamp,portfolio_value\n"
                    "2025-01-01T00:00:00Z,10000\n"
                    "2025-01-01T04:00:00Z,10100\n"
                    "2025-01-01T08:00:00Z,10050\n",
                    encoding="utf-8",
                )
                frame = load_curve("test", path.name)
            finally:
                cfg.MODEL_RESULTS = original
        self.assertIn("cumulative_dsr", frame)
        self.assertIn("expanding_sharpe_ratio", frame)
        self.assertAlmostEqual(frame["return"].iloc[1], 0.01)


if __name__ == "__main__":
    unittest.main()
