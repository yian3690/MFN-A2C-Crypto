"""評估五項資產各 20%、買入後不再平衡的真正 Buy-and-Hold。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_4h as cfg
from src.evaluation_metrics import add_dsr_columns, summarize_dsr


def main() -> None:
    cfg.MODEL_RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(cfg.DATA / "merged_output_test.csv")
    raw["Open Time"] = pd.to_datetime(raw["Open Time"], utc=True)
    if len(raw) != cfg.TEST_ROWS:
        raise ValueError(f"Test 應為 {cfg.TEST_ROWS} 筆，實際為 {len(raw)}。")
    trade = raw.iloc[cfg.LOOKBACK:].reset_index(drop=True)
    initial_per_asset = cfg.INITIAL_BALANCE / len(cfg.PORTFOLIO_ASSETS)
    start_prices = np.array([float(trade.loc[0, f"Open{i}"]) for i in range(4)])
    prices = trade[[f"Open{i}" for i in range(4)]].astype(float).to_numpy()
    crypto_values = prices / start_prices * initial_per_asset
    portfolio_values = crypto_values.sum(axis=1) + initial_per_asset  # USDT
    result = pd.DataFrame({
        "timestamp": trade["Open Time"],
        "portfolio_value": portfolio_values,
    })
    # 四種方法一律由PV_t / PV_{t-1} - 1取得正式評估報酬。
    result["return"] = result["portfolio_value"].pct_change()
    for index, asset in enumerate(cfg.CRYPTO_ASSETS):
        result[f"value_{asset.lower()}"] = crypto_values[:, index]
    result["value_usdt"] = initial_per_asset
    result = add_dsr_columns(
        result,
        eta=cfg.DSR_ETA,
        formula=cfg.EVALUATION_DSR_FORMULA,
    )
    result.to_csv(cfg.MODEL_RESULTS / "buy_and_hold_4h_results.csv", index=False, lineterminator="\n")

    values = result["portfolio_value"]
    returns = values.pct_change().dropna()
    metrics = {
        "Method": "buy_and_hold",
        "DSR Formula": cfg.EVALUATION_DSR_FORMULA,
        "DSR Eta": cfg.DSR_ETA,
        "Initial PV": float(values.iloc[0]),
        "Final PV": float(values.iloc[-1]),
        "Total Return": float(values.iloc[-1] / values.iloc[0] - 1.0),
        "Peak PV": float(values.max()),
        "Max Drawdown": float((values / values.cummax() - 1.0).min()),
        "Sharpe Ratio": float(returns.mean() / returns.std() * np.sqrt(cfg.PERIODS_PER_YEAR)) if returns.std() > 0 else 0.0,
    }
    metrics.update(summarize_dsr(result))
    pd.DataFrame([metrics]).to_csv(cfg.MODEL_RESULTS / "buy_and_hold_4h_metrics.csv", index=False, lineterminator="\n")
    print("=" * 72)
    print("4H EXPERIMENT 1 - STATIC 20% BUY-AND-HOLD")
    print("=" * 72)
    print(f"First trade  : {trade['Open Time'].iloc[0]}")
    print(f"End          : {trade['Open Time'].iloc[-1]}")
    print(f"Initial PV   : {metrics['Initial PV']:.2f}")
    print(f"Final PV     : {metrics['Final PV']:.2f}")
    print(f"Total Return : {metrics['Total Return']:.2%}")
    print(f"Peak PV      : {metrics['Peak PV']:.2f}")
    print(f"Max Drawdown : {metrics['Max Drawdown']:.2%}")
    for index, asset in enumerate(cfg.CRYPTO_ASSETS):
        asset_return = prices[-1, index] / start_prices[index] - 1.0
        print(f"{asset:<4} return : {asset_return:+.2%}")
    print("USDT return : +0.00%")


if __name__ == "__main__":
    main()
