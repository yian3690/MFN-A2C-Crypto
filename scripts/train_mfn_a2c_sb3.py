"""
Train MFN-A2C using Stable-Baselines3.

Before running:
    1. Run download_binance_paper.py
    2. Run prepare_paper_features.py
    3. Run this file.

For the first test, use a small number of timesteps.
After the pipeline works, increase it toward the paper's training setting.
"""



import sys
from pathlib import Path

# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ============================================================
# Imports
# ============================================================

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from src.mfn_sb3_extractor import TwoViewMFN
from src.portfolio_env_sb3 import CryptoPortfolioEnv


DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"


def make_env(training: bool):
    suffix = "_train" if training else "_test"

    env = CryptoPortfolioEnv(
        pct_csv=str(DATA / f"pct_change_output{suffix}.csv"),
        ta_csv=str(DATA / f"ta_test{suffix}.csv"),
        raw_csv=str(DATA / f"merged_output{suffix}.csv"),
        n_previous_timesteps=20,
        max_episode_steps=540 if training else None,
        reward_type="dsr",
        initial_balance=10000,
        eta=0.005,
        random_start=training,
    )

    return Monitor(env)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    train_env = make_env(training=True)

    # Validate Gymnasium interface before training.
    check_env(train_env.unwrapped, warn=True)

    policy_kwargs = dict(
        features_extractor_class=TwoViewMFN,
        features_extractor_kwargs=dict(
            price_dim=16,
            indicator_dim=16,
            lstm_hidden=64,
            memory_dim=128,
            att_hidden=64,
            gate_hidden=64,
            output_dim=128,
            dropout=0.0,
        ),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = A2C(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=7e-4,
        gamma=0.99,
        n_steps=540,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto",
        seed=123,
        tensorboard_log=str(LOGS / "tensorboard"),
    )

    # FIRST TEST:
    # Change this to 10_000 or 100_000 after the pipeline works.
    model.learn(
        total_timesteps=10_000,
        progress_bar=True
    )

    model.save(str(MODELS / "mfn_a2c_test"))

    print("\nTraining finished.")
    print("Saved:", MODELS / "mfn_a2c_test")


if __name__ == "__main__":
    main()
