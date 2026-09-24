"""四種 4H 方法共用的 Test 評估、歸因與訓練診斷。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import A2C

import config_4h as cfg
from src.dman_temporal_attention_extractor import (  # noqa: F401
    DualLSTMDMANTemporalAttention,
)
from src.evaluation_metrics import (
    add_dsr_columns,
    add_strength_alignment_columns,
    print_allocation_summary,
    print_asset_contribution_summary,
    summarize_allocations,
    summarize_asset_contributions,
    summarize_dsr,
    summarize_strength_alignment,
)
from src.price_only_env import PriceOnlyWrapper
from src.training_diagnostics import (
    diagnose_training_file,
    latest_diagnostics,
    print_training_diagnostics_summary,
)
from train_common import METHOD_LABELS

HORIZONS = (3, 6, 18, 42)  # 4H 下分別為 12h、1d、3d、7d
HORIZON_LABELS = ((3, "12h"), (6, "1d"), (18, "3d"), (42, "7d"))


def build_env(method: str):
    base = cfg.make_portfolio_env("test")
    if method == "without_ti":
        return PriceOnlyWrapper(base, price_dim=cfg.PRICE_DIM), base
    return base, base


def attach_timestamps(result: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(raw["Open Time"], utc=True)
    if len(raw) != cfg.TEST_ROWS:
        raise ValueError(f"Test 應為 {cfg.TEST_ROWS} 筆，實際為 {len(raw)}。")
    expected = pd.Timedelta(hours=cfg.BAR_HOURS)
    if not (timestamps.diff().dropna() == expected).all():
        raise ValueError("Test 時間軸不是連續 4H。")
    selected = timestamps.iloc[cfg.LOOKBACK:].reset_index(drop=True)
    if len(selected) != len(result):
        raise ValueError(f"結果 {len(result)} 筆，但時間戳為 {len(selected)} 筆。")
    output = result.copy()
    output.insert(0, "timestamp", selected)
    return output


def print_strength(metrics: dict[str, float]) -> None:
    print("Relative-strength allocation diagnostics:")
    for horizon, label in HORIZON_LABELS:
        corr = metrics[f"{horizon}-step Rank Correlation"]
        winner = metrics[f"{horizon}-step Trailing Winner Weight"]
        loser = metrics[f"{horizon}-step Trailing Loser Weight"]
        print(
            f"  {label:<3}: rank corr {corr:+.3f} | "
            f"winner weight {winner:6.2%} | loser weight {loser:6.2%}"
        )
    print(
        "  Ex-post next-interval winner weight: "
        f"{metrics['Realized Winner Weight']:.2%}"
    )
    print(
        "  Ex-post next-interval loser weight : "
        f"{metrics['Realized Loser Weight']:.2%}"
    )


def evaluate(method: str) -> None:
    if method not in cfg.MODEL_NAMES:
        raise ValueError(f"未知方法：{method}")
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(cfg.DATA / "merged_output_test.csv")
    raw["Open Time"] = pd.to_datetime(raw["Open Time"], utc=True)
    env, base_env = build_env(method)
    model_path = cfg.MODELS / cfg.MODEL_NAMES[method]
    model = A2C.load(str(model_path), env=env, device="cpu")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

    result = attach_timestamps(base_env.get_results(), raw)
    result = add_dsr_columns(
        result,
        eta=base_env.eta,
        warmup_steps=base_env.dsr_warmup_steps,
        formula=base_env.dsr_formula,
    )
    result = add_strength_alignment_columns(result, horizons=HORIZONS)
    result.to_csv(cfg.RESULTS / cfg.RESULT_NAMES[method], index=False)

    values = pd.to_numeric(result["portfolio_value"])
    returns = values.pct_change().dropna()
    metrics = {
        "Method": method,
        "Initial PV": float(values.iloc[0]),
        "Final PV": float(values.iloc[-1]),
        "Total Return": float(values.iloc[-1] / values.iloc[0] - 1.0),
        "Peak PV": float(values.max()),
        "Max Drawdown": float((values / values.cummax() - 1.0).min()),
        "Sharpe Ratio": (
            float(returns.mean() / returns.std() * np.sqrt(cfg.PERIODS_PER_YEAR))
            if returns.std() > 0
            else 0.0
        ),
    }
    metrics.update(summarize_dsr(result))
    allocation = summarize_allocations(result)
    contribution = summarize_asset_contributions(result)
    strength = summarize_strength_alignment(result, horizons=HORIZONS)
    metrics.update(allocation)
    metrics.update(contribution)
    metrics.update(strength)
    metrics_path = cfg.RESULTS / f"{cfg.MODEL_NAMES[method]}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("=" * 72)
    print(f"4H EXPERIMENT 1 - {METHOD_LABELS[method]} BACKTEST")
    print("=" * 72)
    print(f"Model            : {model_path}.zip")
    print(f"Configured steps : {cfg.TOTAL_TIMESTEPS:,} ({cfg.STEP_TAG})")
    print(f"Test data period : {raw['Open Time'].iloc[0]} -> {raw['Open Time'].iloc[-1]}")
    print(f"First trade      : {raw['Open Time'].iloc[cfg.LOOKBACK]}")
    print(f"Initial PV       : {metrics['Initial PV']:.2f}")
    print(f"Final PV         : {metrics['Final PV']:.2f}")
    print(f"Total Return     : {metrics['Total Return']:.2%}")
    print(f"Peak PV          : {metrics['Peak PV']:.2f}")
    print(f"Max Drawdown     : {metrics['Max Drawdown']:.2%}")
    print(f"Sharpe Ratio     : {metrics['Sharpe Ratio']:.4f}")
    print(f"Peak Cum. DSR    : {metrics['Peak Cumulative DSR']:.4f}")
    print(f"Final Cum. DSR   : {metrics['Final Cumulative DSR']:.4f}")
    print_allocation_summary(allocation)
    print_asset_contribution_summary(contribution)
    print_strength(strength)

    diagnostic = latest_diagnostics(
        cfg.LOGS / "training_diagnostics",
        f"{cfg.MODEL_NAMES[method]}_*.csv",
    )
    if diagnostic:
        summary, warnings = diagnose_training_file(diagnostic)
        print_training_diagnostics_summary(diagnostic, summary, warnings)
    env.close()
