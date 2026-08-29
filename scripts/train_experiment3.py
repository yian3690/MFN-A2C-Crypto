"""Train MFN-A2C and A2C with DSR and portfolio-value rewards for Experiment 3."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from src.portfolio_env_sb3 import CryptoPortfolioEnv
from src.mfn_sb3_extractor import TwoViewMFN


DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

MODELS.mkdir(
    parents=True,
    exist_ok=True,
)


TOTAL_TIMESTEPS = 100_000

# 正式實驗：
# TOTAL_TIMESTEPS = 1_800_000


def make_env(reward_type):

    """Create the configured Gymnasium environment used by this script."""
    env = CryptoPortfolioEnv(
        pct_csv=str(
            DATA / "pct_change_output_train.csv"
        ),

        ta_csv=str(
            DATA / "ta_test_train.csv"
        ),

        raw_csv=str(
            DATA / "merged_output_train.csv"
        ),

        n_previous_timesteps=20,

        max_episode_steps=540,

        reward_type=reward_type,

        initial_balance=10000.0,

        eta=0.005,

        random_start=True,
    )

    return Monitor(env)


def train_mfn(reward_type):

    """Train and save the MFN-A2C variant for the selected reward type."""
    env = make_env(
        reward_type
    )

    policy_kwargs = dict(

        features_extractor_class=
            TwoViewMFN,

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

        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64],
        ),
    )

    model = A2C(

        "MlpPolicy",

        env,

        learning_rate=7e-4,

        gamma=0.99,

        n_steps=540,

        policy_kwargs=
            policy_kwargs,

        seed=123,

        verbose=1,

        device="auto",

        tensorboard_log=str(
            LOGS /
            "tensorboard"
        ),
    )

    model.learn(

        total_timesteps=
            TOTAL_TIMESTEPS,

        progress_bar=True,
    )

    output = (

        MODELS /
        f"exp3_mfn_{reward_type}"
    )

    model.save(
        str(output)
    )


def train_a2c(reward_type):

    """Train and save the standard A2C variant for the selected reward type."""
    env = make_env(
        reward_type
    )

    model = A2C(

        "MlpPolicy",

        env,

        learning_rate=7e-4,

        gamma=0.99,

        n_steps=540,

        policy_kwargs=dict(

            net_arch=dict(
                pi=[64, 64],
                vf=[64, 64],
            )
        ),

        seed=123,

        verbose=1,

        device="auto",

        tensorboard_log=str(
            LOGS /
            "tensorboard"
        ),
    )

    model.learn(

        total_timesteps=
            TOTAL_TIMESTEPS,

        progress_bar=True,
    )

    output = (

        MODELS /
        f"exp3_a2c_{reward_type}"
    )

    model.save(
        str(output)
    )


def main():

    """主程式入口：依序執行此腳本定義的完整流程。"""
    print()
    print("==============================")
    print("Experiment 3")
    print("==============================")

    print(
        "Training MFN-A2C + DSR"
    )

    train_mfn("dsr")

    print(
        "Training MFN-A2C + PV"
    )

    train_mfn("pv")

    print(
        "Training A2C + DSR"
    )

    train_a2c("dsr")

    print(
        "Training A2C + PV"
    )

    train_a2c("pv")

    print()
    print(
        "Experiment 3 training completed."
    )


if __name__ == "__main__":
    main()