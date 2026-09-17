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
    VALIDATION_FREQUENCY,
    dqn_algorithm_kwargs,
    make_portfolio_env,
)
from src.validation_callback import ValidationBestModelCallback, read_best_validation


CHECKPOINTS = ROOT / "checkpoints_dqn"


def main():
    """只用Train訓練，定期以Validation Final PV選最佳模型。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    base_env = make_portfolio_env(
        "train",
        action_mode=DQN_ACTION_MODE,
    )
    env = Monitor(
        DiscretePortfolioWrapper(
            base_env,
            weight_step=DQN_WEIGHT_STEP,
            max_crypto_weight=DQN_MAX_CRYPTO_WEIGHT,
        )
    )
    validation_base = make_portfolio_env(
        "validation",
        action_mode=DQN_ACTION_MODE,
    )
    validation_env = Monitor(
        DiscretePortfolioWrapper(
            validation_base,
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
    output = MODELS / DQN_MODEL_NAME
    final_output = MODELS / f"{DQN_MODEL_NAME}_final"
    validation_log = LOGS / "validation" / f"dqn_baseline_{RUN_TAG}.csv"
    validation = ValidationBestModelCallback(
        validation_env,
        eval_freq=VALIDATION_FREQUENCY,
        best_model_path=output,
        log_path=validation_log,
        reset_log=True,
    )

    print("=" * 60)
    print("DQN BASELINE - TRAIN/VALIDATION MODEL SELECTION")
    print("=" * 60)
    print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
    print("Training data   : chronological Train (31,364 rows)")
    print("Validation      : independent 1,080 rows")
    print(f"Validation every: {VALIDATION_FREQUENCY:,} steps")
    print("Model selection : highest Validation Final PV")
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
        callback=[checkpoint, validation],
        progress_bar=True,
    )
    model.save(str(final_output))
    validation.close()
    env.close()
    best_pv, best_step = read_best_validation(validation_log)

    print()
    print("DQN BASELINE VALIDATION SELECTION FINISHED")
    print(f"Best Validation: step={best_step}, Final PV={best_pv:.2f}")
    print(f"Saved best model: {output}.zip")
    print(f"Saved final diagnostic model: {final_output}.zip")


if __name__ == "__main__":
    main()
