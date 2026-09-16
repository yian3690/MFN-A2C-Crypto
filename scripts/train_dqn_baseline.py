"""Train the discrete-action DQN baseline for Experiment 2."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.experiment_periods import LOOKBACK
from src.portfolio_env_sb3 import CryptoPortfolioEnv
from src.discrete_action_env import DiscretePortfolioWrapper


DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
CHECKPOINTS = ROOT / "checkpoints_dqn"

MODELS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)


TOTAL_TIMESTEPS = 600_000
MAX_CRYPTO_WEIGHT = 0.60


def make_env(paths, *, random_start, max_episode_steps):

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

        n_previous_timesteps=LOOKBACK,

        max_episode_steps=max_episode_steps,

        reward_type="dsr",

        initial_balance=10000.0,

        eta=0.005,

        random_start=random_start,
    )


    """
    使用 20% 權重網格。原始五資產完整網格有 126 種配置；套用
    BTC/ETH/LTC/BNB 單一資產最高 60% 後剩下 106 種。USDT 不設上限，
    因此代理仍可在風險升高時選擇 100% USDT。
    """
    env = DiscretePortfolioWrapper(
        env,
        weight_step=0.2,
        max_crypto_weight=MAX_CRYPTO_WEIGHT,
    )

    return Monitor(env)


def main():

    """主程式入口：依序執行此腳本定義的完整流程。"""
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

    print("Observation space :", env.observation_space)
    print("Action space      :", env.action_space)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(CHECKPOINTS),
        name_prefix="DQN",
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,

        learning_rate=7e-4,
        gamma=0.99,

        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,

        train_freq=4,
        gradient_steps=1,

        target_update_interval=10_000,

        exploration_fraction=0.1,
        exploration_final_eps=0.05,

        policy_kwargs=dict(
            net_arch=[64, 64]
        ),

        verbose=1,
        seed=123,

        tensorboard_log=str(
            LOGS / "tensorboard"
        ),

        device="auto",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    output = MODELS / "dqn_baseline"
    model.save(str(output))
    model.save(str(CHECKPOINTS / "DQN_600000_final"))

    print()
    print("Training completed.")
    print(f"Saved model: {output}.zip")

    env.close()


if __name__ == "__main__":
    main()
