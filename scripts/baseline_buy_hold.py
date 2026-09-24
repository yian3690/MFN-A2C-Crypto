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
from src.experiment_periods import LOOKBACK, PERIODS_PER_YEAR
from src.experiment_config import DSR_ETA, DSR_FORMULA


DATA = ROOT / "data"
RESULTS = ROOT / "results"


INITIAL_BALANCE = 10000.0


def main():
    """主程式入口：依序執行此腳本定義的完整流程。"""

    RESULTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        DATA / "merged_output_test.csv"
    )

    test_period = validate_test_period(
        df,
        lookback=LOOKBACK,
    )

    # 與 RL 環境採用相同時間軸：
    # 先完成 observation window，
    # 再由 Open[LOOKBACK] 進場。
    open_columns = [
        "Open0",
        "Open1",
        "Open2",
        "Open3",
    ]

    prices = (
        df[open_columns]
        .astype(float)
        .iloc[LOOKBACK:]
        .reset_index(drop=True)
    )

    initial_prices = prices.iloc[0]

    # Buy-and-hold：
    # BTC / ETH / LTC / BNB / USDT 各配置 20%。
    #
    # 四種加密貨幣會隨價格變動，
    # USDT 視為穩定資產，價值維持不變。
    crypto_weights = np.array([
        0.20,
        0.20,
        0.20,
        0.20,
    ])

    usdt_weight = 0.20

    # 將每個幣的價格除以進場價格。
    # 起始時全部都是 1。
    normalized = (
        prices / initial_prices
    )

    # 四種 cryptocurrency 的資產價值。
    crypto_components = (
        normalized * crypto_weights
    )

    # USDT 價格假設維持 1，
    # 因此它的 normalized value 始終為 1。
    portfolio = (
        crypto_components.sum(axis=1)
        + usdt_weight
    )

    portfolio_value = (
        INITIAL_BALANCE * portfolio
    )

    result = pd.DataFrame({
        "portfolio_value": portfolio_value,
        "return": portfolio_value.pct_change(),
        "turnover": 0.0,
    })

    result = add_test_timestamps(
        result,
        df,
        lookback=LOOKBACK,
    )

    # Buy-and-hold 起始配置：
    #
    # BTC  20%
    # ETH  20%
    # LTC  20%
    # BNB  20%
    # USDT 20%
    #
    # 之後不 rebalance，因此權重會隨資產價格自然漂移。
    for column, asset in zip(
        open_columns,
        ("btc", "eth", "ltc", "bnb"),
    ):
        result[f"weight_{asset}"] = (
            crypto_components[column]
            / portfolio
        )

    # USDT 本身價值不變，
    # 但佔總 portfolio 的比例會隨其他幣價格變化。
    result["weight_usdt"] = (
        usdt_weight / portfolio
    )

    result = add_dsr_columns(
        result,
        eta=DSR_ETA,
        formula=DSR_FORMULA,
    )

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
        portfolio_value.cummax()
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

    returns = (
        result["return"]
        .dropna()
    )

    sharpe = 0.0

    if (
        len(returns) > 1
        and returns.std() > 0
    ):
        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(PERIODS_PER_YEAR)
        )

    print()
    print("=" * 60)
    print("BUY-AND-HOLD BASELINE")
    print(
        "Assets          : "
        "BTC/ETH/LTC/BNB/USDT = 20% each"
    )
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
        f"Peak Cumulative DSR   : "
        f"{dsr_metrics['Peak Cumulative DSR']:.4f}"
    )

    print(
        f"Final Cumulative DSR  : "
        f"{dsr_metrics['Final Cumulative DSR']:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()