"""使用Train資料訓練，並以Validation選擇最佳A2C baseline checkpoint。"""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.experiment_config import (
    A2C_ACTION_MODE,
    A2C_LOG_STD_INIT,
    A2C_NORMALIZE_ADVANTAGE,
    A2C_MODEL_NAME,
    A2C_N_STEPS,
    A2C_UPDATE_EPOCHS,
    CHECKPOINT_FREQUENCY,
    DSR_REWARD_MODE,
    DSR_REWARD_SCALE,
    LOGS,
    LOOKBACK,
    MODELS,
    REWARD_TYPE,
    RETURN_REWARD_SCALE,
    RUN_TAG,
    TOTAL_TIMESTEPS,
    VALIDATION_FREQUENCY,
    a2c_algorithm_kwargs,
    make_portfolio_env,
)
from src.multi_epoch_a2c import MultiEpochA2C
from src.training_diagnostics import TrainingDiagnosticsCallback
from src.validation_callback import ValidationBestModelCallback, read_best_validation


CHECKPOINTS = ROOT / "checkpoints_a2c"


def diagnostics_callback() -> TrainingDiagnosticsCallback:
    """建立單階段訓練的rollout診斷檔。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = (
        LOGS
        / "training_diagnostics"
        / f"a2c_baseline_{RUN_TAG}_{stamp}.csv"
    )
    print(f"Training diagnostics: {path}")
    return TrainingDiagnosticsCallback(path)


def main():
    """在Train訓練，Validation最佳checkpoint直接作為正式模型。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        make_portfolio_env("train", action_mode=A2C_ACTION_MODE)
    )
    validation_env = Monitor(
        make_portfolio_env("validation", action_mode=A2C_ACTION_MODE)
    )
    model = MultiEpochA2C(
        policy="MlpPolicy",
        env=env,
        **a2c_algorithm_kwargs(),
    )
    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(CHECKPOINTS),
        name_prefix=f"A2C_{RUN_TAG.upper()}",
    )
    output = MODELS / A2C_MODEL_NAME
    final_output = MODELS / f"{A2C_MODEL_NAME}_final"
    validation_log = LOGS / "validation" / f"a2c_baseline_{RUN_TAG}.csv"
    validation = ValidationBestModelCallback(
        validation_env,
        eval_freq=VALIDATION_FREQUENCY,
        best_model_path=output,
        log_path=validation_log,
        reset_log=True,
    )

    print("=" * 60)
    print("A2C BASELINE - TRAIN/VALIDATION MODEL SELECTION")
    print("=" * 60)
    print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
    print("Training data   : chronological Train (31,364 rows)")
    print("Validation      : independent 1,080 rows")
    print(f"Validation every: {VALIDATION_FREQUENCY:,} steps")
    print("Model selection : highest Validation Final PV")
    print(f"Window          : {LOOKBACK} ({LOOKBACK * 2} hours)")
    print("Interval        : 2H")
    print("Feature extractor: SB3 MLP")
    print("Action policy   : Gaussian logits -> Softmax")
    print(f"Gaussian log std: {A2C_LOG_STD_INIT:g} (initial std ~= 0.3679)")
    print(f"Normalize advantage: {A2C_NORMALIZE_ADVANTAGE}")
    print(
        f"Reward          : {DSR_REWARD_SCALE:g} x DSR + "
        f"{RETURN_REWARD_SCALE:g} x log return"
    )
    print(f"Reward type     : {REWARD_TYPE}")
    print(f"DSR reward mode : {DSR_REWARD_MODE}")
    print(f"A2C n_steps     : {A2C_N_STEPS}")
    print(f"Update epochs   : {A2C_UPDATE_EPOCHS}")
    print("=" * 60)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=[
            checkpoint,
            diagnostics_callback(),
            validation,
        ],
    )
    model.save(str(final_output))
    validation.close()
    env.close()
    best_pv, best_step = read_best_validation(validation_log)
    if best_step is None:
        raise RuntimeError("沒有產生可用的Validation checkpoint。")

    print()
    print("=" * 60)
    print("A2C BASELINE TRAINING FINISHED")
    print(f"Best Validation: step={best_step}, Final PV={best_pv:.2f}")
    print(f"Saved Validation-best model: {output}.zip")
    print(f"Saved final-step diagnostic model: {final_output}.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
