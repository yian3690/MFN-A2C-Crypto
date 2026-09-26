"""目前唯一4H設定的回歸測試。"""

import sys
import unittest
from pathlib import Path

import numpy as np

from src.dsr import EWMA_CHANGE_FORMULA, PAPER_FORMULA

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import config_4h as cfg


class ExperimentConfigTests(unittest.TestCase):
    def test_shared_4h_budget_and_paths(self):
        self.assertEqual(cfg.BAR_HOURS, 4)
        self.assertGreater(cfg.TOTAL_TIMESTEPS, 0)
        self.assertIsInstance(cfg.SEED, int)
        self.assertEqual(cfg.DSR_FORMULA, PAPER_FORMULA)
        self.assertEqual(cfg.EVALUATION_DSR_FORMULA, EWMA_CHANGE_FORMULA)
        self.assertGreater(cfg.DSR_REWARD_SCALE, 0.0)
        self.assertGreaterEqual(cfg.RETURN_REWARD_SCALE, 0.0)
        self.assertGreater(cfg.LOOKBACK, 0)
        self.assertEqual(cfg.TECHNICAL_INDICATORS, (
            "SMA20", "EMA20", "MACD", "RSI14", "RS_14D",
        ))
        self.assertEqual(cfg.INDICATOR_DIM, 25)
        self.assertIn("rs14d", cfg.RUN_TAG.lower())
        expected_tokens = (
            f"paperdsr{cfg.number_tag(cfg.DSR_REWARD_SCALE)}",
            f"ret{cfg.number_tag(cfg.RETURN_REWARD_SCALE)}",
            f"eta{cfg.number_tag(cfg.DSR_ETA)}",
            f"win{cfg.LOOKBACK}",
            f"logstd{cfg.number_tag(cfg.A2C_LOG_STD_INIT)}",
            cfg.ADVANTAGE_TAG,
            f"e{cfg.A2C_UPDATE_EPOCHS}",
            f"seed{cfg.SEED}",
        )
        for token in expected_tokens:
            self.assertIn(token, cfg.RUN_TAG)
        self.assertEqual(cfg.number_tag(50.0), "50")
        self.assertEqual(cfg.number_tag(0.005), "0p005")
        self.assertEqual(cfg.number_tag(-2), "m2")
        self.assertEqual(cfg.DIAGNOSTICS_EVERY_ROLLOUTS, 5)
        self.assertGreater(cfg.DQN_DIAGNOSTICS_FREQUENCY, 0)
        self.assertIn(f"_{cfg.STEP_TAG}_", cfg.DQN_MODEL_NAME)
        self.assertIn(f"_{cfg.STEP_TAG}_", cfg.A2C_RETURN_MODEL_NAME)
        self.assertIn(f"_{cfg.STEP_TAG}_", cfg.PROPOSED_RETURN_MODEL_NAME)
        self.assertEqual(cfg.DATA, cfg.ROOT / "data")
        self.assertEqual(cfg.MODELS, cfg.ROOT / "models")
        self.assertEqual(cfg.RESULTS, cfg.ROOT / "results")
        self.assertEqual(cfg.MODEL_RESULTS, cfg.RESULTS / "model_result")
        self.assertEqual(
            cfg.EXPERIMENT_RESULTS, cfg.RESULTS / "experiment_result"
        )
        self.assertNotIn("experiment1_4h", str(cfg.DATA))

    def test_data_paths_only_accept_train_and_test(self):
        self.assertTrue(str(cfg.data_paths("train")["raw"]).endswith("merged_output_train.csv"))
        self.assertTrue(str(cfg.data_paths("test")["raw"]).endswith("merged_output_test.csv"))
        for invalid in ("validation", "development", "invalid"):
            with self.assertRaises(ValueError):
                cfg.data_paths(invalid)

    def test_train_is_random_180d_and_test_is_fixed(self):
        train = cfg.make_portfolio_env("train")
        test = cfg.make_portfolio_env("test")
        try:
            self.assertTrue(train.random_start)
            self.assertEqual(train.max_episode_steps, 1_080)
            self.assertFalse(test.random_start)
            self.assertEqual(test.max_episode_steps, cfg.TEST_ROWS - cfg.LOOKBACK - 1)
            self.assertEqual(len(train.raw_data), 15_680)
            self.assertEqual(len(test.raw_data), 1_080)
            self.assertEqual(train.indicator_dim, 25)
            self.assertEqual(test.indicator_dim, 25)
            self.assertIn("BTC_RS_14D", train.ta_data.columns)
            self.assertIn("BTC_RS_14D", test.ta_data.columns)
            self.assertEqual(
                train.return_reward_scale, cfg.RETURN_REWARD_SCALE
            )
        finally:
            train.close(); test.close()

    def test_portfolio_return_reward_is_single_period_return(self):
        env = cfg.make_portfolio_env(
            "test", reward_type="portfolio_return"
        )
        try:
            env.reset()
            _, reward, _, _, info = env.step(
                np.zeros(len(cfg.PORTFOLIO_ASSETS), dtype=np.float32)
            )
            self.assertAlmostEqual(reward, info["portfolio_return"])
            self.assertAlmostEqual(
                reward,
                env.balance_history[-1] / env.balance_history[-2] - 1.0,
            )
            self.assertEqual(info["reward_type"], "portfolio_return")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
