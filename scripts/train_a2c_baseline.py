"""使用32,444筆Train訓練A2C baseline並保存最後一步模型。"""

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
    A2C_MODEL_NAME,
    A2C_N_STEPS,
    A2C_NORMALIZE_ADVANTAGE,
    A2C_UPDATE_EPOCHS,
    CHECKPOINT_FREQUENCY,
    DSR_REWARD_SCALE,
    LOGS,
    LOOKBACK,
    MODELS,
    RETURN_REWARD_SCALE,
    RUN_TAG,
    TOTAL_TIMESTEPS,
    a2c_algorithm_kwargs,
    make_portfolio_env,
)
from src.experiment_periods import EXPECTED_TRAIN_ROWS
from src.multi_epoch_a2c import MultiEpochA2C
from src.training_diagnostics import TrainingDiagnosticsCallback


CHECKPOINTS = ROOT / "checkpoints_a2c"


def diagnostics_callback() -> TrainingDiagnosticsCallback:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOGS / "training_diagnostics" / f"a2c_baseline_{RUN_TAG}_{stamp}.csv"
    print(f"Training diagnostics: {path}")
    return TrainingDiagnosticsCallback(path)


def main():
    """固定訓練預算，不使用Validation或Test選模。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    env = Monitor(make_portfolio_env("train", action_mode=A2C_ACTION_MODE))
    model = MultiEpochA2C("MlpPolicy", env, **a2c_algorithm_kwargs())
    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(CHECKPOINTS),
        name_prefix=f"A2C_{RUN_TAG.upper()}",
    )
    output = MODELS / A2C_MODEL_NAME

    print("=" * 64)
    print("A2C BASELINE - FIXED FINAL TRAINING")
    print("=" * 64)
    print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"Training rows   : {EXPECTED_TRAIN_ROWS:,}")
    print("Episode         : complete chronological Train")
    print("Model selection : fixed final-step model; no Validation")
    print(f"Window          : {LOOKBACK} ({LOOKBACK * 2} hours)")
    print(f"Gaussian log std: {A2C_LOG_STD_INIT:g}")
    print(f"Normalize advantage: {A2C_NORMALIZE_ADVANTAGE}")
    print(f"Reward          : {DSR_REWARD_SCALE:g} x DSR + {RETURN_REWARD_SCALE:g} x log return")
    print(f"A2C n_steps     : {A2C_N_STEPS}")
    print(f"Update epochs   : {A2C_UPDATE_EPOCHS}")
    print("=" * 64)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=[checkpoint, diagnostics_callback()],
    )
    model.save(str(output))
    env.close()
    print(f"Saved fixed final model: {output}.zip")


if __name__ == "__main__":
    main()
