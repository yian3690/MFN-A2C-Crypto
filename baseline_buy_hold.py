from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


INITIAL_BALANCE = 10000.0


def main():

    df = pd.read_csv(
        DATA / "merged_output_test.csv"
    )

    # 4 cryptocurrencies
    close_columns = [
        "Close0",
        "Close1",
        "Close2",
        "Close3",
    ]

    prices = df[close_columns].astype(float)

    initial_prices = prices.iloc[0]

    # Equal-weight BTC/ETH/LTC/BNB
    weights = np.array(
        [0.25, 0.25, 0.25, 0.25]
    )

    normalized = (
        prices / initial_prices
    )

    portfolio = (
        normalized * weights
    ).sum(axis=1)

    portfolio_value = (
        INITIAL_BALANCE
        * portfolio
    )

    result = pd.DataFrame({
        "portfolio_value":
            portfolio_value
    })

    result.to_csv(
        ROOT / "buy_hold_results.csv",
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

    print()
    print("=" * 60)
    print("BUY-AND-HOLD BASELINE")
    print("=" * 60)

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
        f"Max Drawdown    : {max_drawdown:.2f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()