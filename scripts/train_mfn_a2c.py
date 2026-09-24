"""使用32,444筆Train訓練MFN-A2C並保存最後一步模型。"""

import argparse
import math
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
    A2C_N_STEPS,
    A2C_NORMALIZE_ADVANTAGE,
    A2C_UPDATE_EPOCHS,
    CHECKPOINT_FREQUENCY,
    DSR_REWARD_SCALE,
    LOGS,
    LOOKBACK,
    MFN_DEVICE,
    MFN_DIAGNOSTICS_EVERY_ROLLOUTS,
    MFN_MODEL_NAME,
    MFN_RESUME_TAG,
    MFN_RUN_TAG,
    MODELS,
    RETURN_REWARD_SCALE,
    TOTAL_TIMESTEPS,
    a2c_algorithm_kwargs,
    a2c_policy_kwargs,
    make_portfolio_env,
)
from src.experiment_periods import EXPECTED_TRAIN_ROWS
from src.feature_schema import INDICATOR_DIM, PRICE_DIM
from src.mfn_github_extractor import GitHubStyleTwoViewMFN
from src.multi_epoch_a2c import MultiEpochA2C
from src.training_diagnostics import TrainingDiagnosticsCallback


CHECKPOINTS = ROOT / "checkpoints"
CHECKPOINT_PREFIX = f"MFN_A2C_RESUME_{MFN_RESUME_TAG.upper()}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"從最新相容checkpoint接續到總計{TOTAL_TIMESTEPS:,}步。",
    )
    return parser.parse_args()


def checkpoint_timesteps(path: str | Path, prefix: str = CHECKPOINT_PREFIX):
    """從固定最後一步實驗的checkpoint檔名取得累計步數。"""
    match = re.fullmatch(
        rf"{re.escape(prefix)}_(\d+)_steps\.zip",
        Path(path).name,
    )
    return int(match.group(1)) if match else None


def find_latest_checkpoint(
    directory: str | Path,
    prefix: str = CHECKPOINT_PREFIX,
    max_timesteps: int | None = None,
) -> Path | None:
    """尋找不超過目標步數的最新相容checkpoint。"""
    candidates = []
    for path in Path(directory).glob(f"{prefix}_*_steps.zip"):
        timesteps = checkpoint_timesteps(path, prefix)
        if timesteps is not None and (
            max_timesteps is None or timesteps <= max_timesteps
        ):
            candidates.append((timesteps, path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def policy_kwargs() -> dict:
    """建立MFN與Actor/Critic網路。"""
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
    kwargs = a2c_algorithm_kwargs()
    kwargs["device"] = MFN_DEVICE
    kwargs["policy_kwargs"] = policy_kwargs()
    return MultiEpochA2C("MlpPolicy", env, **kwargs)


def diagnostics_callback(completed: int = 0) -> TrainingDiagnosticsCallback:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = (
        LOGS
        / "training_diagnostics"
        / f"mfn_a2c_{MFN_RUN_TAG}_from_{completed}_{stamp}.csv"
    )
    return TrainingDiagnosticsCallback(
        path,
        record_every_rollouts=MFN_DIAGNOSTICS_EVERY_ROLLOUTS,
    )


def main():
    """固定訓練預算，可續訓，但不使用Validation或Test選模。"""
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    (LOGS / "tensorboard").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    env = Monitor(make_portfolio_env("train", action_mode=A2C_ACTION_MODE))

    resume_path = (
        find_latest_checkpoint(CHECKPOINTS, max_timesteps=TOTAL_TIMESTEPS)
        if args.resume
        else None
    )
    if args.resume and resume_path is None:
        raise FileNotFoundError(
            "--resume找不到相容checkpoint；已停止以避免誤從零開始。"
        )
    if resume_path is None:
        model = new_model(env)
        completed = 0
    else:
        model = MultiEpochA2C.load(
            str(resume_path),
            env=env,
            device=MFN_DEVICE,
            tensorboard_log=str(LOGS / "tensorboard"),
        )
        completed = int(model.num_timesteps)
    remaining = max(TOTAL_TIMESTEPS - completed, 0)
    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(CHECKPOINTS),
        name_prefix=CHECKPOINT_PREFIX,
    )
    output = MODELS / MFN_MODEL_NAME

    print("=" * 64)
    print("MFN-A2C - FIXED FINAL TRAINING")
    print("=" * 64)
    print(f"Target steps    : {TOTAL_TIMESTEPS:,}")
    print(f"Resume source   : {resume_path if resume_path else 'none'}")
    print(f"Completed       : {completed:,}")
    print(f"Remaining       : {remaining:,}")
    print(f"Training rows   : {EXPECTED_TRAIN_ROWS:,}")
    print("Episode         : complete chronological Train")
    print("Model selection : fixed final-step model; no Validation")
    print(f"Window          : {LOOKBACK} ({LOOKBACK * 2} hours)")
    print(f"Gaussian log std: {A2C_LOG_STD_INIT:g} (std ~= {math.exp(A2C_LOG_STD_INIT):.4f})")
    print(f"Normalize advantage: {A2C_NORMALIZE_ADVANTAGE}")
    print(f"Reward          : {DSR_REWARD_SCALE:g} x DSR + {RETURN_REWARD_SCALE:g} x log return")
    print(f"A2C n_steps     : {A2C_N_STEPS}")
    print(f"Update epochs   : {A2C_UPDATE_EPOCHS}")
    print(f"Training device : {MFN_DEVICE}")
    print("=" * 64)

    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=resume_path is None,
            progress_bar=True,
            callback=[checkpoint, diagnostics_callback(completed)],
        )
    model.save(str(output))
    env.close()
    print(f"Saved fixed final model: {output}.zip")


if __name__ == "__main__":
    main()
