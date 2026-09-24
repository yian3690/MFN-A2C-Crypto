"""三種4H A2C方法共用的訓練與跨目標步數續訓流程。"""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import config_4h as cfg
from src.dman_temporal_attention_extractor import DualLSTMDMANTemporalAttention
from src.multi_epoch_a2c import MultiEpochA2C
from src.price_only_env import PriceOnlyWrapper
from src.training_diagnostics import TrainingDiagnosticsCallback

METHOD_LABELS = {
    "a2c": "A2C BASELINE",
    "without_ti": "A2C WITHOUT TI",
    "dman_attention": "DUAL-LSTM + DMAN + TEMPORAL SELF-ATTENTION A2C",
}
_TARGET_STEP_PATTERN = re.compile(r"_\d+K_", flags=re.IGNORECASE)
_CHECKPOINT_STEPS_PATTERN = re.compile(r"_(\d+)_steps\.zip$", flags=re.IGNORECASE)


def build_env(split: str, method: str):
    base = cfg.make_portfolio_env(split)
    if method == "without_ti":
        return Monitor(PriceOnlyWrapper(base, price_dim=cfg.PRICE_DIM))
    return Monitor(base)



def model_kwargs(method: str) -> dict:
    """依方法指定特徵擷取器，其餘A2C條件完全共用。"""
    kwargs = cfg.a2c_algorithm_kwargs()
    policy_kwargs = cfg.a2c_policy_kwargs()

    if method == "dman_attention":
        # 保留雙LSTM與DMAN，並以跨時間注意力取代MGM。
        policy_kwargs.update(
            features_extractor_class=DualLSTMDMANTemporalAttention,
            features_extractor_kwargs={
                "price_dim": cfg.PRICE_DIM,
                "indicator_dim": cfg.INDICATOR_DIM,
                "lstm_hidden": 64,
                "dman_hidden": 64,
                "temporal_dim": 128,
                "temporal_attention_heads": 4,
                "dropout": 0.0,
            },
        )
        kwargs["policy_kwargs"] = policy_kwargs
        kwargs["device"] = cfg.ATTENTION_DEVICE
    return kwargs


def normalized_model_name(method: str) -> str:
    """移除目標總步數，使300k與600k可判定為同一實驗設定。"""
    if method not in cfg.MODEL_NAMES:
        raise ValueError(f"未知方法：{method}")
    return _TARGET_STEP_PATTERN.sub("_", cfg.MODEL_NAMES[method].upper(), count=1)


def checkpoint_prefix(method: str) -> str:
    """新checkpoint採不含目標總步數的固定prefix。"""
    return f"{normalized_model_name(method)}_RESUME"


def checkpoint_timesteps(path: str | Path) -> int | None:
    match = _CHECKPOINT_STEPS_PATTERN.search(Path(path).name)
    return int(match.group(1)) if match else None


def _checkpoint_is_compatible(path: str | Path, method: str) -> bool:
    """接受新固定prefix，也向下相容舊的300K／600K prefix。"""
    name = Path(path).name
    match = _CHECKPOINT_STEPS_PATTERN.search(name)
    if not match:
        return False
    saved_prefix = name[: match.start()]
    if saved_prefix.upper() == checkpoint_prefix(method).upper():
        return True
    return (
        _TARGET_STEP_PATTERN.sub("_", saved_prefix.upper(), count=1)
        == normalized_model_name(method)
    )


def find_latest_checkpoint(
    directory: str | Path,
    method: str,
    max_timesteps: int | None = None,
) -> Path | None:
    """尋找設定相容且不超過目前目標步數的最新checkpoint。"""
    candidates: list[tuple[int, Path]] = []
    for path in Path(directory).glob("*.zip"):
        timesteps = checkpoint_timesteps(path)
        if (
            timesteps is not None
            and _checkpoint_is_compatible(path, method)
            and (max_timesteps is None or timesteps <= max_timesteps)
        ):
            candidates.append((timesteps, path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def parse_args(method: str, argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=f"訓練{METHOD_LABELS[method]}")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"從最新相容checkpoint接續到總計{cfg.TOTAL_TIMESTEPS:,}步。",
    )
    return parser.parse_args(argv)


def train(method: str, argv: list[str] | None = None) -> None:
    """固定最後步模型；可由舊目標步數checkpoint接續訓練。"""
    if method not in METHOD_LABELS:
        raise ValueError(f"未知方法：{method}")
    args = parse_args(method, argv)

    method_checkpoints = cfg.CHECKPOINTS / method
    for directory in (
        cfg.MODELS,
        cfg.LOGS / "tensorboard",
        cfg.LOGS / "training_diagnostics",
        method_checkpoints,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    env = build_env("train", method)
    resume_path = (
        find_latest_checkpoint(
            method_checkpoints,
            method,
            max_timesteps=cfg.TOTAL_TIMESTEPS,
        )
        if args.resume
        else None
    )
    if args.resume and resume_path is None:
        env.close()
        raise FileNotFoundError(
            "--resume找不到相容checkpoint；已停止以避免誤從零開始。"
        )

    kwargs = model_kwargs(method)
    if resume_path is None:
        model = MultiEpochA2C(policy="MlpPolicy", env=env, **kwargs)
        completed = 0
    else:
        model = MultiEpochA2C.load(
            str(resume_path),
            env=env,
            device=kwargs.get("device", "auto"),
            tensorboard_log=str(cfg.LOGS / "tensorboard"),
        )
        completed = int(model.num_timesteps)

    remaining = max(cfg.TOTAL_TIMESTEPS - completed, 0)
    model_name = cfg.MODEL_NAMES[method]
    output = cfg.MODELS / model_name
    checkpoint = CheckpointCallback(
        save_freq=cfg.CHECKPOINT_FREQUENCY,
        save_path=str(method_checkpoints),
        name_prefix=checkpoint_prefix(method),
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostics_path = (
        cfg.LOGS
        / "training_diagnostics"
        / f"{model_name}_from_{completed}_{stamp}.csv"
    )
    diagnostics = TrainingDiagnosticsCallback(
        diagnostics_path,
        record_every_rollouts=cfg.DIAGNOSTICS_EVERY_ROLLOUTS,
    )

    print("=" * 72)
    print(f"4H EXPERIMENT 1 - {METHOD_LABELS[method]}")
    print("=" * 72)
    print(f"Target steps    : {cfg.TOTAL_TIMESTEPS:,}")
    print(f"Resume source   : {resume_path if resume_path else 'none'}")
    print(f"Completed       : {completed:,}")
    print(f"Remaining       : {remaining:,}")
    print(f"Run/model tag   : {cfg.STEP_TAG} (由TOTAL_TIMESTEPS自動產生)")
    print(f"Interval        : {cfg.BAR_HOURS}H")
    print(f"Window          : {cfg.LOOKBACK} ({cfg.LOOKBACK * cfg.BAR_HOURS} hours)")
    print(
        f"Train episode   : {cfg.TRAIN_EPISODE_STEPS} bars "
        f"({cfg.TRAIN_EPISODE_DAYS} days), random start"
    )
    print(f"Test            : {cfg.TEST_ROWS} bars ({cfg.TEST_ROWS / cfg.BARS_PER_DAY:.0f} days)")
    print(f"RS_14D          : {cfg.RELATIVE_STRENGTH_BARS} bars")
    print("Validation      : disabled")
    print(
        f"Reward          : {cfg.DSR_REWARD_SCALE:g} x DSR + "
        f"{cfg.RETURN_REWARD_SCALE:g} x log return"
    )
    print(
        f"Gaussian std    : exp({cfg.A2C_LOG_STD_INIT}) = "
        f"{math.exp(cfg.A2C_LOG_STD_INIT):.4f}"
    )
    print(f"Normalize adv.  : {cfg.A2C_NORMALIZE_ADVANTAGE}")
    print(f"A2C n_steps     : {cfg.A2C_N_STEPS} ({cfg.A2C_N_STEPS / cfg.BARS_PER_DAY:.0f} days)")
    print(f"Update epochs   : {cfg.A2C_UPDATE_EPOCHS}")
    print(f"Diagnostics     : every {cfg.DIAGNOSTICS_EVERY_ROLLOUTS} rollouts")
    print("Model selection : final training model (no Validation)")
    print("=" * 72)

    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=resume_path is None,
            progress_bar=True,
            callback=[checkpoint, diagnostics],
        )
    model.save(str(output))
    env.close()
    print(f"Saved final model: {output}.zip")
