"""使用完整Train資料，單階段訓練不含技術指標的A2C。"""

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
    A2C_UPDATE_EPOCHS,
    A2C_WITHOUT_TI_MODEL_NAME,
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
from src.feature_schema import PRICE_DIM
from src.multi_epoch_a2c import MultiEpochA2C
from src.price_only_env import PriceOnlyWrapper
from src.training_diagnostics import TrainingDiagnosticsCallback


CHECKPOINTS = ROOT / "checkpoints_a2c_without_ti"


def diagnostics_callback() -> TrainingDiagnosticsCallback:
    """建立A2C w/o TI單階段rollout診斷檔。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = (
        LOGS
        / "training_diagnostics"
        / f"a2c_without_ti_{RUN_TAG}_{stamp}.csv"
    )
    print(f"Training diagnostics: {path}")
    return TrainingDiagnosticsCallback(path)


def main():
    """在完整Development上訓練固定300k步，Test保持隔離。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    base_env = make_portfolio_env(
        "development",
        action_mode=A2C_ACTION_MODE,
    )
    env = Monitor(PriceOnlyWrapper(base_env, price_dim=PRICE_DIM))
    model = MultiEpochA2C(
        policy="MlpPolicy",
        env=env,
        **a2c_algorithm_kwargs(),
    )
    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(CHECKPOINTS),
        name_prefix=f"A2C_WITHOUT_TI_{RUN_TAG.upper()}",
    )

    print("=" * 60)
    print("A2C WITHOUT TI - SINGLE-STAGE FULL-TRAIN TRAINING")
    print("=" * 60)
    print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
    print("Training data   : complete chronological Train")
    print("Validation      : disabled")
    print("Model selection : fixed final 300k model")
    print(f"Gaussian log std: {A2C_LOG_STD_INIT:g} (initial std ~= 0.3679)")
    print(f"Normalize advantage: {A2C_NORMALIZE_ADVANTAGE}")
    print(f"Window          : {LOOKBACK} ({LOOKBACK * 2} hours)")
    print(f"Price features  : {PRICE_DIM}")
    print("Technical features: 0")
    print(
        f"Reward          : {DSR_REWARD_SCALE:g} x DSR + "
        f"{RETURN_REWARD_SCALE:g} x log return"
    )
    print(f"Update epochs   : {A2C_UPDATE_EPOCHS}")
    print("=" * 60)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=[checkpoint, diagnostics_callback()],
    )
    output = MODELS / A2C_WITHOUT_TI_MODEL_NAME
    model.save(str(output))
    env.close()

    print()
    print("A2C WITHOUT TI SINGLE-STAGE TRAINING FINISHED")
    print(f"Saved final model: {output}.zip")


if __name__ == "__main__":
    main()
