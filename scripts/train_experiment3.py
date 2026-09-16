"""Train MFN-A2C and A2C with DSR and portfolio-value rewards for Experiment 3."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3.common.monitor import Monitor

from src.multi_epoch_a2c import MultiEpochA2C
from src.portfolio_env_sb3 import CryptoPortfolioEnv
from src.mfn_sb3_extractor import TwoViewMFN
from src.feature_schema import INDICATOR_DIM, PRICE_DIM


DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

MODELS.mkdir(
    parents=True,
    exist_ok=True,
)


TOTAL_TIMESTEPS = 600_000
UPDATE_EPOCHS = 18


def make_env(paths, reward_type):

    """Create the configured Gymnasium environment used by this script."""
    env = CryptoPortfolioEnv(
        pct_csv=str(
            paths["pct"]
        ),

        ta_csv=str(
            paths["ta"]
        ),

        raw_csv=str(
            paths["raw"]
        ),

        n_previous_timesteps=20,

        max_episode_steps=540,

        reward_type=reward_type,

        initial_balance=10000.0,

        eta=0.005,

        random_start=True,
    )

    return Monitor(env)


def train_mfn(paths, reward_type):

    """Train and save the MFN-A2C variant for the selected reward type."""
    env = make_env(
        paths,
        reward_type,
    )

    policy_kwargs = dict(

        features_extractor_class=
            TwoViewMFN,

        features_extractor_kwargs=dict(

            price_dim=PRICE_DIM,
            indicator_dim=INDICATOR_DIM,

            lstm_hidden=64,

            memory_dim=128,

            output_dim=128,

            dropout=0.0,
        ),

        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64],
        ),
    )

    model = MultiEpochA2C(

        "MlpPolicy",

        env,

        learning_rate=7e-4,

        gamma=0.99,

        n_steps=540,

        update_epochs=UPDATE_EPOCHS,

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


def train_a2c(paths, reward_type):

    """Train and save the standard A2C variant for the selected reward type."""
    env = make_env(
        paths,
        reward_type,
    )

    model = MultiEpochA2C(

        "MlpPolicy",

        env,

        learning_rate=7e-4,

        gamma=0.99,

        n_steps=540,

        update_epochs=UPDATE_EPOCHS,

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
    train_paths = {
        "pct": DATA / "pct_change_output_train.csv",
        "ta": DATA / "ta_test_train.csv",
        "raw": DATA / "merged_output_train.csv",
    }
    print()
    print("==============================")
    print("Experiment 3")
    print("==============================")

    print(
        "Training MFN-A2C + DSR"
    )

    train_mfn(train_paths, "dsr")

    print(
        "Training MFN-A2C + PV"
    )

    train_mfn(train_paths, "pv")

    print(
        "Training A2C + DSR"
    )

    train_a2c(train_paths, "dsr")

    print(
        "Training A2C + PV"
    )

    train_a2c(train_paths, "pv")

    print()
    print(
        "Experiment 3 training completed."
    )


if __name__ == "__main__":
    main()
