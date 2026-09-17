"""使用完整Train資料，單階段訓練離散動作DQN baseline。"""

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


CHECKPOINTS = ROOT / "checkpoints_dqn"


def main():
    """在完整Development上訓練固定300k步，Test保持隔離。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    base_env = make_portfolio_env(
        "development",
        action_mode=DQN_ACTION_MODE,
    )
    env = Monitor(
        DiscretePortfolioWrapper(
            base_env,
            weight_step=DQN_WEIGHT_STEP,
            max_crypto_weight=DQN_MAX_CRYPTO_WEIGHT,
        )
    )
    model = DQN(
        policy="MlpPolicy",
        env=env,
        **dqn_algorithm_kwargs(),
    )
    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(CHECKPOINTS),
        name_prefix=f"DQN_{RUN_TAG.upper()}",
    )

    print("=" * 60)
    print("DQN BASELINE - SINGLE-STAGE FULL-TRAIN TRAINING")
    print("=" * 60)
    print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
    print("Training data   : complete chronological Train")
    print("Validation      : disabled")
    print("Model selection : fixed final 300k model")
    print(f"Window          : {LOOKBACK} ({LOOKBACK * 2} hours)")
    print("Observation space:", env.observation_space)
    print("Action space     :", env.action_space)
    print(
        f"Reward          : {DSR_REWARD_SCALE:g} x DSR + "
        f"{RETURN_REWARD_SCALE:g} x log return"
    )
    print("=" * 60)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint,
        progress_bar=True,
    )
    output = MODELS / DQN_MODEL_NAME
    model.save(str(output))
    env.close()

    print()
    print("DQN BASELINE SINGLE-STAGE TRAINING FINISHED")
    print(f"Saved final model: {output}.zip")


if __name__ == "__main__":
    main()
