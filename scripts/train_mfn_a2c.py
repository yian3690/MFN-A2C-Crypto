"""使用Train資料訓練，並以Validation選擇最佳MFN-A2C checkpoint。"""

import argparse
import re
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
    A2C_N_STEPS,
    A2C_UPDATE_EPOCHS,
    CHECKPOINT_FREQUENCY,
    DSR_REWARD_SCALE,
    LOGS,
    LOOKBACK,
    MFN_MODEL_NAME,
    MODELS,
    RETURN_REWARD_SCALE,
    RUN_TAG,
    TOTAL_TIMESTEPS,
    VALIDATION_FREQUENCY,
    a2c_algorithm_kwargs,
    a2c_policy_kwargs,
    make_portfolio_env,
)
from src.feature_schema import INDICATOR_DIM, PRICE_DIM
from src.mfn_github_extractor import GitHubStyleTwoViewMFN
from src.multi_epoch_a2c import MultiEpochA2C
from src.training_diagnostics import TrainingDiagnosticsCallback
from src.validation_callback import ValidationBestModelCallback, read_best_validation


CHECKPOINTS = ROOT / "checkpoints"
MODEL_NAME = MFN_MODEL_NAME
CHECKPOINT_PREFIX = f"MFN_A2C_{RUN_TAG.upper()}"


def parse_args():
    """解析是否從本實驗最新相容checkpoint接續訓練。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="從最新相容checkpoint接續到總計300,000步。",
    )
    return parser.parse_args()


def checkpoint_timesteps(
    path: str | Path,
    prefix: str = CHECKPOINT_PREFIX,
) -> int | None:
    """從相容checkpoint檔名擷取累計步數。"""
    match = re.fullmatch(
        rf"{re.escape(prefix)}_(\d+)_steps\.zip",
        Path(path).name,
    )
    return int(match.group(1)) if match else None


def find_latest_checkpoint(
    directory: str | Path,
    prefix: str = CHECKPOINT_PREFIX,
) -> Path | None:
    """尋找本實驗累計步數最大的相容checkpoint。"""
    candidates = []
    for path in Path(directory).glob(f"{prefix}_*_steps.zip"):
        timesteps = checkpoint_timesteps(path, prefix)
        if timesteps is not None:
            candidates.append((timesteps, path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def policy_kwargs() -> dict:
    """建立MFN特徵擷取器與Actor/Critic網路設定。"""
    kwargs = a2c_policy_kwargs()
    kwargs.update(
        features_extractor_class=GitHubStyleTwoViewMFN,
        features_extractor_kwargs=dict(
            price_dim=PRICE_DIM,
            indicator_dim=INDICATOR_DIM,
            lstm_hidden=64,
            memory_dim=128,
            attention_hidden=64,
            candidate_hidden=64,
            gate_hidden=64,
            dropout=0.0,
        ),
    )
    return kwargs


def new_model(env) -> MultiEpochA2C:
    """以共用A2C超參數建立MFN-A2C。"""
    kwargs = a2c_algorithm_kwargs()
    kwargs["policy_kwargs"] = policy_kwargs()
    return MultiEpochA2C(policy="MlpPolicy", env=env, **kwargs)


def diagnostics_callback(completed: int = 0) -> TrainingDiagnosticsCallback:
    """建立MFN單階段訓練的診斷CSV。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = (
        LOGS
        / "training_diagnostics"
        / f"mfn_a2c_{RUN_TAG}_from_{completed}_{stamp}.csv"
    )
    return TrainingDiagnosticsCallback(path)


def main():
    """在Train訓練，Validation最佳checkpoint直接作為正式模型。"""
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        make_portfolio_env("train", action_mode=A2C_ACTION_MODE)
    )
    validation_env = Monitor(
        make_portfolio_env("validation", action_mode=A2C_ACTION_MODE)
    )
    resume_path = (
        find_latest_checkpoint(CHECKPOINTS) if args.resume else None
    )
    if resume_path is not None:
        model = MultiEpochA2C.load(
            str(resume_path),
            env=env,
            device="auto",
            tensorboard_log=str(LOGS / "tensorboard"),
        )
        completed = int(model.num_timesteps)
    else:
        model = new_model(env)
        completed = 0

    remaining = max(TOTAL_TIMESTEPS - completed, 0)
    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(CHECKPOINTS),
        name_prefix=CHECKPOINT_PREFIX,
    )
    output = MODELS / MODEL_NAME
    final_output = MODELS / f"{MODEL_NAME}_final"
    validation_log = LOGS / "validation" / f"mfn_a2c_{RUN_TAG}.csv"
    validation = ValidationBestModelCallback(
        validation_env,
        eval_freq=VALIDATION_FREQUENCY,
        best_model_path=output,
        log_path=validation_log,
        reset_log=not (args.resume and resume_path is not None),
    )

    print("=" * 60)
    print("MFN-A2C - TRAIN/VALIDATION MODEL SELECTION")
    print("=" * 60)
    print(f"Target steps    : {TOTAL_TIMESTEPS:,}")
    print(f"Completed       : {completed:,}")
    print(f"Remaining       : {remaining:,}")
    print("Training data   : chronological Train (31,364 rows)")
    print("Validation      : independent 1,080 rows")
    print(f"Validation every: {VALIDATION_FREQUENCY:,} steps")
    print("Model selection : highest Validation Final PV")
    print(f"Gaussian log std: {A2C_LOG_STD_INIT:g} (initial std ~= 0.3679)")
    print(f"Normalize advantage: {A2C_NORMALIZE_ADVANTAGE}")
    print(f"Window          : {LOOKBACK} ({LOOKBACK * 2} hours)")
    print(
        f"Reward          : {DSR_REWARD_SCALE:g}x DSR + "
        f"{RETURN_REWARD_SCALE:g}x log return"
    )
    print(f"A2C n_steps     : {A2C_N_STEPS}")
    print(f"Update epochs   : {A2C_UPDATE_EPOCHS}")
    print("=" * 60)

    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=not (args.resume and resume_path),
            progress_bar=True,
            callback=[
                checkpoint,
                diagnostics_callback(completed),
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
    print("MFN-A2C TRAINING FINISHED")
    print(f"Best Validation: step={best_step}, Final PV={best_pv:.2f}")
    print(f"Saved Validation-best model: {output}.zip")
    print(f"Saved final-step diagnostic model: {final_output}.zip")


if __name__ == "__main__":
    main()
