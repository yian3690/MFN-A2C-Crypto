"""共用公平比較設定的回歸測試。"""

import unittest

import numpy as np

from src.dsr import PAPER_FORMULA
from src.experiment_config import (
    A2C_ACTION_MODE,
    A2C_LOG_STD_INIT,
    A2C_NORMALIZE_ADVANTAGE,
    A2C_N_STEPS,
    A2C_UPDATE_EPOCHS,
    DQN_ACTION_MODE,
    DSR_FORMULA,
    DSR_REWARD_SCALE,
    DSR_REWARD_MODE,
    FEATURE_VARIANT,
    REWARD_TYPE,
    RETURN_REWARD_SCALE,
    RUN_TAG,
    SEED,
    TOTAL_TIMESTEPS,
    VALIDATION_FREQUENCY,
    a2c_policy_kwargs,
    a2c_algorithm_kwargs,
    data_paths,
    make_portfolio_env,
)
from src.experiment_periods import LOOKBACK


class ExperimentConfigTests(unittest.TestCase):
    def test_shared_fair_comparison_budget(self):
        self.assertEqual(TOTAL_TIMESTEPS, 300_000)
        self.assertEqual(VALIDATION_FREQUENCY, 100_000)
        self.assertEqual(SEED, 123)
        self.assertEqual(DSR_FORMULA, PAPER_FORMULA)
        self.assertEqual(DSR_REWARD_SCALE, 200.0)
        self.assertEqual(RETURN_REWARD_SCALE, 50.0)
        self.assertEqual(REWARD_TYPE, "hybrid")
        self.assertEqual(DSR_REWARD_MODE, "step")
        self.assertEqual(A2C_N_STEPS, 540)
        self.assertEqual(A2C_UPDATE_EPOCHS, 1)
        self.assertEqual(A2C_LOG_STD_INIT, -1.0)
        self.assertTrue(A2C_NORMALIZE_ADVANTAGE)
        self.assertTrue(a2c_algorithm_kwargs()["normalize_advantage"])
        self.assertEqual(a2c_policy_kwargs()["log_std_init"], -1.0)
        self.assertEqual(FEATURE_VARIANT, "level_zscore_rs14d")
        self.assertIn("300k", RUN_TAG)
        self.assertIn("valselect", RUN_TAG)
        self.assertIn("hybrid_dsr200_ret50", RUN_TAG)
        self.assertIn("win20", RUN_TAG)
        self.assertIn("logstdm1", RUN_TAG)
        self.assertIn("normadv", RUN_TAG)
        self.assertEqual(LOOKBACK, 20)
        self.assertIn("level_zscore_rs14d", RUN_TAG)
        self.assertIn("val1080_pv100k", RUN_TAG)

    def test_data_paths_are_split_consistently(self):
        train = data_paths("train")
        validation = data_paths("validation")
        development = data_paths("development")
        test = data_paths("test")
        self.assertTrue(str(train["raw"]).endswith("merged_output_train.csv"))
        self.assertTrue(
            str(validation["raw"]).endswith("merged_output_validation.csv")
        )
        self.assertTrue(
            str(development["raw"]).endswith("merged_output_development.csv")
        )
        self.assertTrue(str(test["raw"]).endswith("merged_output_test.csv"))
        with self.assertRaises(ValueError):
            data_paths("invalid")

    def test_train_environments_are_chronological(self):
        a2c_env = make_portfolio_env("train", action_mode=A2C_ACTION_MODE)
        dqn_env = make_portfolio_env("train", action_mode=DQN_ACTION_MODE)
        try:
            self.assertFalse(a2c_env.random_start)
            self.assertFalse(dqn_env.random_start)
            self.assertEqual(len(a2c_env.raw_data), 31_364)
            self.assertEqual(len(dqn_env.raw_data), 31_364)
            self.assertEqual(a2c_env.action_mode, "logits")
            self.assertEqual(dqn_env.action_mode, "simplex")
            self.assertEqual(a2c_env.dsr_formula, PAPER_FORMULA)
            self.assertEqual(dqn_env.dsr_formula, PAPER_FORMULA)
            self.assertEqual(a2c_env.dsr_reward_scale, 200.0)
            self.assertEqual(dqn_env.dsr_reward_scale, 200.0)
            self.assertEqual(a2c_env.return_reward_scale, 50.0)
            self.assertEqual(dqn_env.return_reward_scale, 50.0)
            self.assertEqual(a2c_env.n_previous_timesteps, 20)
            self.assertEqual(dqn_env.n_previous_timesteps, 20)
            self.assertEqual(a2c_env.reward_type, "hybrid")
            self.assertEqual(dqn_env.reward_type, "hybrid")
            self.assertEqual(a2c_env.dsr_reward_mode, "step")
            self.assertEqual(dqn_env.dsr_reward_mode, "step")
        finally:
            a2c_env.close()
            dqn_env.close()

    def test_validation_is_fixed_and_separate_from_test(self):
        validation_env = make_portfolio_env(
            "validation",
            action_mode=A2C_ACTION_MODE,
        )
        test_env = make_portfolio_env("test", action_mode=A2C_ACTION_MODE)
        try:
            self.assertFalse(validation_env.random_start)
            self.assertFalse(test_env.random_start)
            self.assertEqual(len(validation_env.raw_data), 1080)
            self.assertEqual(len(test_env.raw_data), 1080)
            self.assertLess(
                validation_env.raw_data["Open Time"].iloc[-1],
                test_env.raw_data["Open Time"].iloc[0],
            )
        finally:
            validation_env.close()
            test_env.close()

    def test_reward_is_hybrid_but_reported_dsr_is_raw(self):
        env = make_portfolio_env("train", action_mode=A2C_ACTION_MODE)
        try:
            env.reset(seed=SEED)
            observed_nonzero_dsr = False
            for _ in range(10):
                reward, info = None, None
                _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
                expected_dsr = DSR_REWARD_SCALE * info["DSR"]
                expected_return = (
                    RETURN_REWARD_SCALE
                    * np.log1p(info["portfolio_return"])
                )
                self.assertAlmostEqual(
                    reward,
                    expected_dsr + expected_return,
                    places=12,
                )
                self.assertAlmostEqual(
                    info["scaled_DSR_reward"],
                    expected_dsr,
                    places=12,
                )
                self.assertAlmostEqual(
                    info["scaled_return_reward"],
                    expected_return,
                    places=12,
                )
                observed_nonzero_dsr |= not np.isclose(info["DSR"], 0.0)
            self.assertTrue(observed_nonzero_dsr)

            result = env.get_results()
            assets = ("btc", "eth", "ltc", "bnb", "usdt")
            step_contribution = result[
                [f"return_contribution_{asset}" for asset in assets]
            ].sum(axis=1, min_count=1)
            np.testing.assert_allclose(
                step_contribution.dropna().to_numpy(),
                result["return"].dropna().to_numpy(),
                rtol=1e-12,
                atol=1e-12,
            )
            attributed_pnl = result[
                [f"pnl_contribution_{asset}" for asset in assets]
            ].sum().sum()
            actual_pnl = (
                result["portfolio_value"].iloc[-1]
                - result["portfolio_value"].iloc[0]
            )
            self.assertAlmostEqual(attributed_pnl, actual_pnl, places=8)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
