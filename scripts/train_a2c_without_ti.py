<<<<<<< Updated upstream
"""Train the price-only A2C ablation for Experiment 1."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.experiment_periods import LOOKBACK
from src.feature_schema import PRICE_DIM
from src.multi_epoch_a2c import MultiEpochA2C
from src.portfolio_env_sb3 import CryptoPortfolioEnv
from src.price_only_env import PriceOnlyWrapper


DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
CHECKPOINTS = ROOT / "checkpoints_a2c_without_ti"

TOTAL_TIMESTEPS = 600_000
UPDATE_EPOCHS = 18


def make_env(paths, *, random_start, max_episode_steps):

    """Create the configured Gymnasium environment used by this script."""
    base_env = CryptoPortfolioEnv(
        pct_csv=str(
            paths["pct"]
        ),

        ta_csv=str(
            paths["ta"]
        ),

        raw_csv=str(
            paths["raw"]
        ),

        n_previous_timesteps=LOOKBACK,

        max_episode_steps=max_episode_steps,

        reward_type="dsr",

        eta=0.005,

        initial_balance=10000,

        random_start=random_start,
    )

    # Remove technical indicators.
    env = PriceOnlyWrapper(
        base_env,
        price_dim=PRICE_DIM,
    )

    return Monitor(env)


def main():
    """主程式入口：依序執行此腳本定義的完整流程。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    train_paths = {
        "pct": DATA / "pct_change_output_train.csv",
        "ta": DATA / "ta_test_train.csv",
        "raw": DATA / "merged_output_train.csv",
    }
    env = make_env(
        train_paths,
        random_start=True,
        max_episode_steps=540,
    )

    print("=" * 60)
    print("A2C WITHOUT TECHNICAL INDICATORS")
    print("=" * 60)

    print(
        f"Observation shape : {env.observation_space.shape}"
    )

    print(
        f"Total timesteps   : {TOTAL_TIMESTEPS}"
    )

    print(f"Price features    : {PRICE_DIM}")
    print("Technical features: 0")
    print("MFN               : NO")
    print("Historical window : 20")
    print("Time interval     : 2H")
    print("DSR eta           : 0.005")
    print("A2C gamma         : 0.99")
    print("A2C n_steps       : 540")
    print(f"Update epochs     : {UPDATE_EPOCHS}")
    print("Learning rate     : 7e-4")

    print("=" * 60)

    policy_kwargs = dict(

        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64],
        )
    )

    model = MultiEpochA2C(

        policy="MlpPolicy",

        env=env,

        learning_rate=7e-4,

        gamma=0.99,

        n_steps=540,

        update_epochs=UPDATE_EPOCHS,

        policy_kwargs=policy_kwargs,

        verbose=1,

        device="auto",

        seed=123,

        tensorboard_log=str(
            LOGS / "tensorboard"
        ),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(CHECKPOINTS),
        name_prefix="A2C_without_TI",
    )

    model.learn(

        total_timesteps=TOTAL_TIMESTEPS,

        progress_bar=True,

        callback=checkpoint_callback,
    )

    output = MODELS / "a2c_without_ti"
    model.save(str(output))
    model.save(str(CHECKPOINTS / "A2C_without_TI_600000_final"))

    print()
    print("=" * 60)
    print("A2C WITHOUT TI TRAINING FINISHED")
    print("=" * 60)

    env.close()

=======
"""訓練 4H Experiment 1 A2C without technical indicators。"""
from train_common import train
>>>>>>> Stashed changes

if __name__ == "__main__":
    train("without_ti")
