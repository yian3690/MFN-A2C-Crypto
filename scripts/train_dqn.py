"""Train Experiment 2 DQN on 4H data, with explicit checkpoint resume."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import config_4h as cfg
from src.discrete_action_env import DiscretePortfolioWrapper
from src.dqn_training_diagnostics import DQNTrainingDiagnosticsCallback

_TARGET_STEP_PATTERN = re.compile(r"_\d+K_", flags=re.IGNORECASE)
_CHECKPOINT_STEPS_PATTERN = re.compile(r"_(\d+)_steps\.zip$", flags=re.IGNORECASE)


def normalized_model_name() -> str:
    """移除目標總步數，讓 300k checkpoint 可被 600k/900k 設定辨識。"""
    return _TARGET_STEP_PATTERN.sub("_", cfg.DQN_MODEL_NAME.upper(), count=1)


def checkpoint_prefix() -> str:
    return f"{normalized_model_name()}_RESUME"


def checkpoint_timesteps(path: str | Path) -> int | None:
    match = _CHECKPOINT_STEPS_PATTERN.search(Path(path).name)
    return int(match.group(1)) if match else None


def _compatible(path: Path) -> bool:
    match = _CHECKPOINT_STEPS_PATTERN.search(path.name)
    if not match:
        return False
    saved_prefix = path.name[:match.start()].upper()
    return saved_prefix == checkpoint_prefix().upper() or (
        _TARGET_STEP_PATTERN.sub("_", saved_prefix, count=1)
        == normalized_model_name()
    )


def find_latest_checkpoint(directory: str | Path, max_timesteps: int) -> Path | None:
    candidates = []
    for path in Path(directory).glob("*.zip"):
        steps = checkpoint_timesteps(path)
        if steps is not None and steps <= max_timesteps and _compatible(path):
            candidates.append((steps, path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def replay_buffer_path(checkpoint: str | Path) -> Path:
    """對應 SB3 CheckpointCallback 的 replay-buffer 檔名。"""
    path = Path(checkpoint)
    match = _CHECKPOINT_STEPS_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"無法辨識 checkpoint 步數：{path}")
    prefix = path.name[:match.start()]
    steps = match.group(1)
    return path.with_name(f"{prefix}_replay_buffer_{steps}_steps.pkl")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="訓練 4H Experiment 2 DQN")
    parser.add_argument(
        "--resume", action="store_true",
        help=f"從最新相容 checkpoint 接續到 {cfg.TOTAL_TIMESTEPS:,} 步。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint_dir = cfg.CHECKPOINTS / "dqn"
    for directory in (
        cfg.MODELS, cfg.LOGS / "tensorboard",
        cfg.LOGS / "training_diagnostics", checkpoint_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    base_env = cfg.make_portfolio_env("train", action_mode=cfg.DQN_ACTION_MODE)
    env = Monitor(DiscretePortfolioWrapper(
        base_env,
        weight_step=cfg.DQN_WEIGHT_STEP,
        max_crypto_weight=cfg.DQN_MAX_CRYPTO_WEIGHT,
    ))
    resume_path = (
        find_latest_checkpoint(checkpoint_dir, cfg.TOTAL_TIMESTEPS)
        if args.resume else None
    )
    if args.resume and resume_path is None:
        env.close()
        raise FileNotFoundError(
            "--resume 找不到設定相容的 DQN checkpoint；已停止以避免誤從零開始。"
        )

    if resume_path is None:
        model = DQN("MlpPolicy", env, **cfg.dqn_algorithm_kwargs())
        completed = 0
    else:
        model = DQN.load(
            str(resume_path), env=env, device="auto",
            tensorboard_log=str(cfg.LOGS / "tensorboard"),
        )
        completed = int(model.num_timesteps)
        replay_path = replay_buffer_path(resume_path)
        if replay_path.exists():
            model.load_replay_buffer(str(replay_path))
            print(f"Loaded replay buffer: {replay_path}")
        else:
            print("WARNING: checkpoint 沒有 replay buffer；模型參數會續訓，但經驗池從空白開始。")

    remaining = max(cfg.TOTAL_TIMESTEPS - completed, 0)
    checkpoint = CheckpointCallback(
        save_freq=cfg.CHECKPOINT_FREQUENCY,
        save_path=str(checkpoint_dir),
        name_prefix=checkpoint_prefix(),
        save_replay_buffer=True,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostics_path = (
        cfg.LOGS / "training_diagnostics"
        / f"{cfg.DQN_MODEL_NAME}_from_{completed}_{stamp}.csv"
    )
    diagnostics = DQNTrainingDiagnosticsCallback(
        diagnostics_path,
        record_every_steps=cfg.DQN_DIAGNOSTICS_FREQUENCY,
    )

    print("=" * 72)
    print("4H EXPERIMENT 2 - DQN BASELINE")
    print("=" * 72)
    print(f"Target steps    : {cfg.TOTAL_TIMESTEPS:,}")
    print(f"Resume source   : {resume_path if resume_path else 'none'}")
    print(f"Completed       : {completed:,}")
    print(f"Remaining       : {remaining:,}")
    print(f"Run/model tag   : {cfg.DQN_RUN_TAG}")
    print(f"Train episode   : {cfg.TRAIN_EPISODE_DAYS} days, random start")
    print(f"Window          : {cfg.LOOKBACK} ({cfg.LOOKBACK * cfg.BAR_HOURS} hours)")
    print(f"Discrete actions: {env.action_space.n}")
    print(f"Train frequency : every {cfg.DQN_TRAIN_FREQUENCY} environment steps")
    print(f"Gradient steps  : {cfg.DQN_GRADIENT_STEPS} per update")
    print(f"Learning rate   : {cfg.DQN_LEARNING_RATE:g}")
    print(f"Diagnostics     : every {cfg.DQN_DIAGNOSTICS_FREQUENCY:,} steps")
    print("Validation      : disabled")
    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=resume_path is None,
            callback=[checkpoint, diagnostics],
            progress_bar=True,
        )
    output = cfg.MODELS / cfg.DQN_MODEL_NAME
    model.save(str(output))
    model.save_replay_buffer(str(output) + "_replay_buffer.pkl")
    env.close()
    print(f"Saved final model: {output}.zip")


if __name__ == "__main__":
    main()
