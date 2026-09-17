"""使用完整Train資料，以單階段流程訓練A2C baseline。"""

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
    a2c_algorithm_kwargs,
    make_portfolio_env,
)
from src.multi_epoch_a2c import MultiEpochA2C
from src.training_diagnostics import TrainingDiagnosticsCallback


CHECKPOINTS = ROOT / "checkpoints_a2c"


def diagnostics_callback() -> TrainingDiagnosticsCallback:
    """建立本次單階段訓練的rollout診斷檔。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = (
        LOGS
        / "training_diagnostics"
        / f"a2c_baseline_{RUN_TAG}_{stamp}.csv"
    )
    print(f"Training diagnostics: {path}")
    return TrainingDiagnosticsCallback(path)


def main():
    """在完整Development（原Train＋Validation）上訓練固定300k步。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    # development正是論文所稱的完整Train；獨立Test仍完全保留。
    env = Monitor(
        make_portfolio_env("development", action_mode=A2C_ACTION_MODE)
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

    print("=" * 60)
    print("A2C BASELINE - SINGLE-STAGE FULL-TRAIN TRAINING")
    print("=" * 60)
    print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
    print("Training data   : complete chronological Train")
    print("Validation      : disabled")
    print("Model selection : fixed final 300k model")
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
        callback=[checkpoint, diagnostics_callback()],
    )
    output = MODELS / A2C_MODEL_NAME
    model.save(str(output))
    env.close()

    print()
    print("=" * 60)
    print("A2C BASELINE SINGLE-STAGE TRAINING FINISHED")
    print(f"Saved final model: {output}.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
