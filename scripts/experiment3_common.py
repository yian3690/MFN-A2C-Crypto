"""4H Experiment 3的PV reward訓練與評估共用流程。"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config_4h as cfg
from evaluate_common import attach_timestamps, print_strength
from src.evaluation_metrics import (
    add_dsr_columns, add_strength_alignment_columns,
    print_allocation_summary, print_asset_contribution_summary,
    summarize_allocations, summarize_asset_contributions,
    summarize_dsr, summarize_strength_alignment,
)
from src.multi_epoch_a2c import MultiEpochA2C
from src.training_diagnostics import TrainingDiagnosticsCallback

_STEP_PATTERN = re.compile(r"_\d+k_", re.IGNORECASE)


def _names(method: str) -> tuple[str, str]:
    if method == "a2c":
        return cfg.A2C_PV_MODEL_NAME, cfg.A2C_PV_RESULT_NAME
    raise ValueError(f"未知方法：{method}")


def _model_kwargs(method: str) -> dict:
    if method != "a2c":
        raise ValueError(f"未知方法：{method}")
    return cfg.a2c_algorithm_kwargs()


def train(method: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    model_name, _ = _names(method)
    checkpoint_dir = cfg.CHECKPOINTS / f"{method}_pv"
    for directory in (cfg.MODELS, cfg.LOGS / "tensorboard",
                      cfg.LOGS / "training_diagnostics", checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    env = Monitor(cfg.make_portfolio_env("train", reward_type="pv"))
    prefix = _STEP_PATTERN.sub("_", model_name, count=1).upper() + "_RESUME"
    candidates = []
    for path in checkpoint_dir.glob(f"{prefix}_*_steps.zip"):
        match = re.search(r"_(\d+)_steps\.zip$", path.name)
        if match and int(match.group(1)) <= cfg.TOTAL_TIMESTEPS:
            candidates.append((int(match.group(1)), path))
    resume_path = max(candidates, default=(0, None))[1] if args.resume else None
    if args.resume and resume_path is None:
        env.close(); raise FileNotFoundError("找不到相容PV checkpoint。")
    kwargs = _model_kwargs(method)
    if resume_path:
        model = MultiEpochA2C.load(str(resume_path), env=env,
                                   device=kwargs.get("device", "auto"),
                                   tensorboard_log=str(cfg.LOGS / "tensorboard"))
        completed = int(model.num_timesteps)
    else:
        model = MultiEpochA2C("MlpPolicy", env, **kwargs); completed = 0
    checkpoint = CheckpointCallback(save_freq=cfg.CHECKPOINT_FREQUENCY,
                                    save_path=str(checkpoint_dir), name_prefix=prefix)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostics = TrainingDiagnosticsCallback(
        cfg.LOGS / "training_diagnostics" / f"{model_name}_from_{completed}_{stamp}.csv",
        record_every_rollouts=cfg.DIAGNOSTICS_EVERY_ROLLOUTS,
    )
    remaining = max(cfg.TOTAL_TIMESTEPS - completed, 0)
    print(f"4H EXPERIMENT 3 - {method.upper()} + ABSOLUTE PV REWARD")
    print(f"Completed: {completed:,}; remaining: {remaining:,}")
    if remaining:
        model.learn(total_timesteps=remaining, reset_num_timesteps=resume_path is None,
                    progress_bar=True, callback=[checkpoint, diagnostics])
    model.save(str(cfg.MODELS / model_name)); env.close()
    print(f"Saved final model: {cfg.MODELS / model_name}.zip")


def evaluate(method: str) -> None:
    model_name, result_name = _names(method)
    cfg.MODEL_RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(cfg.DATA / "merged_output_test.csv")
    env = cfg.make_portfolio_env("test", reward_type="pv")
    model = A2C.load(str(cfg.MODELS / model_name), env=env, device="cpu")
    obs, _ = env.reset(); done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
    result = attach_timestamps(env.get_results(), raw)
    result = add_dsr_columns(result, eta=env.eta,
                             warmup_steps=env.dsr_warmup_steps,
                             formula=env.dsr_formula)
    result = add_strength_alignment_columns(result, horizons=(3, 6, 18, 42))
    result.to_csv(cfg.MODEL_RESULTS / result_name, index=False)
    values = pd.to_numeric(result["portfolio_value"]); returns = values.pct_change().dropna()
    metrics = {
        "Method": f"{method}_pv", "Initial PV": float(values.iloc[0]),
        "Final PV": float(values.iloc[-1]),
        "Total Return": float(values.iloc[-1] / values.iloc[0] - 1),
        "Peak PV": float(values.max()),
        "Max Drawdown": float((values / values.cummax() - 1).min()),
        "Sharpe Ratio": float(returns.mean() / returns.std() * np.sqrt(cfg.PERIODS_PER_YEAR)) if returns.std() > 0 else 0.0,
    }
    allocation = summarize_allocations(result); contribution = summarize_asset_contributions(result)
    strength = summarize_strength_alignment(result, horizons=(3, 6, 18, 42))
    metrics.update(summarize_dsr(result)); metrics.update(allocation)
    metrics.update(contribution); metrics.update(strength)
    pd.DataFrame([metrics]).to_csv(
        cfg.MODEL_RESULTS / result_name.replace("_results.csv", "_metrics.csv"), index=False)
    print("=" * 72); print(f"4H EXPERIMENT 3 - {method.upper()} + PV BACKTEST"); print("=" * 72)
    print(f"Final PV     : {metrics['Final PV']:.2f}")
    print(f"Total Return : {metrics['Total Return']:.2%}")
    print(f"Peak PV      : {metrics['Peak PV']:.2f}")
    print(f"Max Drawdown : {metrics['Max Drawdown']:.2%}")
    print(f"Sharpe Ratio : {metrics['Sharpe Ratio']:.4f}")
    print_allocation_summary(allocation); print_asset_contribution_summary(contribution)
    print_strength(strength); env.close()
