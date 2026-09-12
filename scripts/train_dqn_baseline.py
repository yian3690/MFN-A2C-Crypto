"""Train the discrete-action DQN baseline for Experiment 2."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.portfolio_env_sb3 import CryptoPortfolioEnv
from src.discrete_action_env import DiscretePortfolioWrapper


DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
CHECKPOINTS = ROOT / "checkpoints_dqn"

MODELS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)


TOTAL_TIMESTEPS = 6_000_00
# Formal experiment:
# TOTAL_TIMESTEPS = 1_800_000


def make_env():

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

        reward_type="dsr",

        initial_balance=10000.0,

        eta=0.005,

        random_start=True,
    )


    """
    為什麼選 20%，不是 10%？
    因為 10% 的步長會產生 100 個離散動作，這會導致 DQN 訓練過程中需要更多的探索和學習時間，可能會增加訓練的複雜性和不穩定性。
    而 20% 的步長會產生 126 個離散動作，這樣的動作空間相對較小，更容易讓 DQN 學習到有效的策略，並且在訓練過程中更穩定。
    """
    env = DiscretePortfolioWrapper(env)

    return Monitor(env)


def main():

    """主程式入口：依序執行此腳本定義的完整流程。"""
    env = make_env()

    print("Observation space :", env.observation_space)
    print("Action space      :", env.action_space)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(CHECKPOINTS),
        name_prefix="dqn",
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

    print()
    print("Training completed.")
    print(f"Saved model: {output}.zip")


if __name__ == "__main__":
    main()