"""只含Train/Test之公平比較設定回歸測試。"""

import unittest

import numpy as np

from src.dsr import PAPER_FORMULA
from src.experiment_config import (
    A2C_ACTION_MODE,
    A2C_LOG_STD_INIT,
    A2C_N_STEPS,
    A2C_NORMALIZE_ADVANTAGE,
    A2C_UPDATE_EPOCHS,
    DQN_ACTION_MODE,
    DSR_FORMULA,
    DSR_REWARD_MODE,
    DSR_REWARD_SCALE,
    FEATURE_VARIANT,
    MFN_DEVICE,
    MFN_DIAGNOSTICS_EVERY_ROLLOUTS,
    MFN_RESUME_TAG,
    MFN_RUN_TAG,
    REWARD_TYPE,
    RETURN_REWARD_SCALE,
    RUN_COMPAT_TAG,
    RUN_TAG,
    SEED,
    TOTAL_TIMESTEPS,
    a2c_algorithm_kwargs,
    a2c_policy_kwargs,
    data_paths,
    make_portfolio_env,
)
from src.experiment_periods import EXPECTED_TRAIN_ROWS, LOOKBACK, TEST_ROWS


class ExperimentConfigTests(unittest.TestCase):
    def test_shared_fair_comparison_budget(self):
        self.assertEqual(TOTAL_TIMESTEPS, 300_000)
        self.assertEqual(SEED, 456)
        self.assertEqual(DSR_FORMULA, PAPER_FORMULA)
        self.assertEqual(DSR_REWARD_SCALE, 1.0)
        self.assertEqual(RETURN_REWARD_SCALE, 50.0)
        self.assertEqual(REWARD_TYPE, "hybrid")
        self.assertEqual(DSR_REWARD_MODE, "step")
        self.assertEqual(A2C_N_STEPS, 540)
        self.assertEqual(A2C_UPDATE_EPOCHS, 1)
        self.assertEqual(A2C_LOG_STD_INIT, -2.0)
        self.assertTrue(A2C_NORMALIZE_ADVANTAGE)
        self.assertTrue(a2c_algorithm_kwargs()["normalize_advantage"])
        self.assertEqual(a2c_policy_kwargs()["log_std_init"], -2.0)
        self.assertEqual(FEATURE_VARIANT, "level_zscore_rs14d")
        self.assertIn("fixedfinal", RUN_TAG)
        self.assertIn("300k", RUN_TAG)
        self.assertIn("train32444", RUN_TAG)
        self.assertNotIn("val", RUN_TAG)
        self.assertEqual(LOOKBACK, 20)
        self.assertEqual(MFN_DEVICE, "cpu")
        self.assertEqual(MFN_DIAGNOSTICS_EVERY_ROLLOUTS, 5)
        self.assertEqual(MFN_RUN_TAG, f"{RUN_TAG}_cpu_diag5")
        self.assertNotIn("300k", RUN_COMPAT_TAG)
        self.assertIn(RUN_COMPAT_TAG, MFN_RESUME_TAG)

    def test_data_paths_only_accept_train_and_test(self):
        train = data_paths("train")
        test = data_paths("test")
        self.assertTrue(str(train["raw"]).endswith("merged_output_train.csv"))
        self.assertTrue(str(test["raw"]).endswith("merged_output_test.csv"))
        for invalid in ("validation", "development", "invalid"):
            with self.assertRaises(ValueError):
                data_paths(invalid)

    def test_train_and_test_environments_are_chronological(self):
        a2c_env = make_portfolio_env("train", action_mode=A2C_ACTION_MODE)
        dqn_env = make_portfolio_env("train", action_mode=DQN_ACTION_MODE)
        test_env = make_portfolio_env("test", action_mode=A2C_ACTION_MODE)
        try:
            self.assertFalse(a2c_env.random_start)
            self.assertFalse(dqn_env.random_start)
            self.assertFalse(test_env.random_start)
            self.assertEqual(len(a2c_env.raw_data), EXPECTED_TRAIN_ROWS)
            self.assertEqual(len(dqn_env.raw_data), EXPECTED_TRAIN_ROWS)
            self.assertEqual(len(test_env.raw_data), TEST_ROWS)
            self.assertEqual(a2c_env.action_mode, "logits")
            self.assertEqual(dqn_env.action_mode, "simplex")
            self.assertEqual(a2c_env.dsr_formula, PAPER_FORMULA)
            self.assertEqual(a2c_env.dsr_reward_scale, 1.0)
            self.assertEqual(a2c_env.return_reward_scale, 50.0)
            self.assertEqual(a2c_env.reward_type, "hybrid")
            self.assertEqual(a2c_env.dsr_reward_mode, "step")
        finally:
            a2c_env.close()
            dqn_env.close()
            test_env.close()

    def test_reward_is_hybrid_but_reported_dsr_is_raw(self):
        env = make_portfolio_env("train", action_mode=A2C_ACTION_MODE)
        try:
            env.reset(seed=SEED)
            observed_nonzero_dsr = False
            for _ in range(10):
                _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
                expected_dsr = DSR_REWARD_SCALE * info["DSR"]
                expected_return = RETURN_REWARD_SCALE * np.log1p(
                    info["portfolio_return"]
                )
                self.assertAlmostEqual(
                    reward, expected_dsr + expected_return, places=12
                )
                observed_nonzero_dsr |= not np.isclose(info["DSR"], 0.0)
            self.assertTrue(observed_nonzero_dsr)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
