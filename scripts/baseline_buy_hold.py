"""Buy-and-hold baseline for the paper experiments."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_metrics import (
    add_test_timestamps,
    add_dsr_columns,
    print_test_period,
    summarize_dsr,
    validate_test_period,
)
from src.experiment_periods import PERIODS_PER_YEAR


DATA = ROOT / "data"
RESULTS = ROOT / "results"


INITIAL_BALANCE = 10000.0


def main():
    """主程式入口：依序執行此腳本定義的完整流程。"""
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        DATA / "merged_output_test.csv"
    )
    test_period = validate_test_period(df, lookback=20)

    # 與 RL 環境採用相同時間軸：先觀察 20 根完整 K 線，再由 Open[20] 進場。
    open_columns = [
        "Open0",
        "Open1",
        "Open2",
        "Open3",
    ]

    prices = (
        df[open_columns]
        .astype(float)
        .iloc[20:]
        .reset_index(drop=True)
    )

    initial_prices = prices.iloc[0]

    # Paper definition: equally allocate to the four cryptocurrencies.
    # USDT remains part of the RL action space but is excluded from B&H.
    crypto_weights = np.array(
        [0.25, 0.25, 0.25, 0.25]
    )
    usdt_weight = 0.0

    normalized = (
        prices / initial_prices
    )

    crypto_components = normalized * crypto_weights
    portfolio = crypto_components.sum(axis=1) + usdt_weight

    portfolio_value = INITIAL_BALANCE * portfolio

    result = pd.DataFrame({
        "portfolio_value": portfolio_value,
        "return": portfolio_value.pct_change(),
        "turnover": 0.0,
    })
    result = add_test_timestamps(result, df, lookback=20)

    # Buy-and-hold starts at 25% per crypto, then weights drift with prices.
    for column, asset in zip(
        open_columns,
        ("btc", "eth", "ltc", "bnb"),
    ):
        result[f"weight_{asset}"] = crypto_components[column] / portfolio
    result["weight_usdt"] = usdt_weight / portfolio

    result = add_dsr_columns(result)
    dsr_metrics = summarize_dsr(result)

    result.to_csv(
        RESULTS / "buy_hold_results.csv",
        index=False,
    )

    initial = portfolio_value.iloc[0]
    final = portfolio_value.iloc[-1]

    total_return = (
        final / initial - 1
    ) * 100

    peak = portfolio_value.max()

    running_max = (
        portfolio_value
        .cummax()
    )

    drawdown = (
        portfolio_value
        / running_max
        - 1
    )

    max_drawdown = (
        drawdown.min()
        * 100
    )

    returns = result["return"].dropna()
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR)

    print()
    print("=" * 60)
    print("BUY-AND-HOLD BASELINE")
    print("Assets          : BTC/ETH/LTC/BNB = 25% each; USDT = 0%")
    print("=" * 60)
    print_test_period(test_period)

    print(
        f"Initial PV      : {initial:.2f}"
    )

    print(
        f"Final PV        : {final:.2f}"
    )

    print(
        f"Peak PV         : {peak:.2f}"
    )

    print(
        f"Total Return    : {total_return:.2f}%"
    )

    print(
        f"Max Drawdown          : {max_drawdown:.2f}%"
    )

    print(
        f"Sharpe Ratio          : {sharpe:.4f}"
    )

    print(
        f"Peak Cumulative DSR   : {dsr_metrics['Peak Cumulative DSR']:.4f}"
    )

    print(
        f"Final Cumulative DSR  : {dsr_metrics['Final Cumulative DSR']:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
