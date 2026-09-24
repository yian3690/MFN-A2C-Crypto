"""使用32,444筆Train訓練DQN baseline並保存最後一步模型。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.discrete_action_env import DiscretePortfolioWrapper
from src.experiment_config import (
    CHECKPOINT_FREQUENCY,
    DQN_ACTION_MODE,
    DQN_MAX_CRYPTO_WEIGHT,
    DQN_MODEL_NAME,
    DQN_WEIGHT_STEP,
    DSR_REWARD_SCALE,
    LOGS,
    LOOKBACK,
    MODELS,
    RETURN_REWARD_SCALE,
    RUN_TAG,
    TOTAL_TIMESTEPS,
    dqn_algorithm_kwargs,
    make_portfolio_env,
)
from src.experiment_periods import EXPECTED_TRAIN_ROWS


CHECKPOINTS = ROOT / "checkpoints_dqn"


def main():
    """固定訓練預算，不使用Validation或Test選模。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    base_env = make_portfolio_env("train", action_mode=DQN_ACTION_MODE)
    env = Monitor(
        DiscretePortfolioWrapper(
            base_env,
            weight_step=DQN_WEIGHT_STEP,
            max_crypto_weight=DQN_MAX_CRYPTO_WEIGHT,
        )
    )
    model = DQN("MlpPolicy", env, **dqn_algorithm_kwargs())
    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(CHECKPOINTS),
        name_prefix=f"DQN_{RUN_TAG.upper()}",
    )
    output = MODELS / DQN_MODEL_NAME

    print("=" * 64)
    print("DQN BASELINE - FIXED FINAL TRAINING")
    print("=" * 64)
    print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"Training rows   : {EXPECTED_TRAIN_ROWS:,}")
    print("Episode         : complete chronological Train")
    print("Model selection : fixed final-step model; no Validation")
    print(f"Window          : {LOOKBACK} ({LOOKBACK * 2} hours)")
    print("Observation space:", env.observation_space)
    print("Action space     :", env.action_space)
    print(f"Reward          : {DSR_REWARD_SCALE:g} x DSR + {RETURN_REWARD_SCALE:g} x log return")
    print("=" * 64)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint,
        progress_bar=True,
    )
    model.save(str(output))
    env.close()
    print(f"Saved fixed final model: {output}.zip")


if __name__ == "__main__":
    main()
