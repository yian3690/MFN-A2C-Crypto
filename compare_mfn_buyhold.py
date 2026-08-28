from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

INITIAL_BALANCE = 10000.0
LOOKBACK = 20


def calculate_buy_and_hold(raw):
    """
    Equal-weight Buy-and-Hold:
    BTC 25%
    ETH 25%
    LTC 25%
    BNB 25%

    Skip the first 20 rows so that the comparison starts
    at the same effective point as the MFN-A2C environment.
    """

    close_columns = [
        "Close0",
        "Close1",
        "Close2",
        "Close3",
    ]

    prices = raw[close_columns].astype(float)

    # Same effective start as MFN-A2C
    prices = prices.iloc[LOOKBACK:].reset_index(drop=True)

    # Equal-weight portfolio
    weights = np.array([
        0.25,
        0.25,
        0.25,
        0.25,
    ])

    initial_prices = prices.iloc[0]

    normalized_prices = (
        prices / initial_prices
    )

    portfolio_value = (
        INITIAL_BALANCE
        * (normalized_prices * weights).sum(axis=1)
    )

    return portfolio_value


def main():

    # ------------------------------------------------------------
    # 1. Load MFN-A2C results
    # ------------------------------------------------------------

    mfn_file = ROOT / "formal_backtest_results.csv"

    if not mfn_file.exists():
        raise FileNotFoundError(
            "找不到 formal_backtest_results.csv"
        )

    mfn = pd.read_csv(mfn_file)

    if "portfolio_value" not in mfn.columns:
        raise ValueError(
            "formal_backtest_results.csv 缺少 portfolio_value 欄位"
        )

    mfn_value = (
        mfn["portfolio_value"]
        .astype(float)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # 2. Load raw test data
    # ------------------------------------------------------------

    raw_file = DATA / "merged_output_test.csv"

    if not raw_file.exists():
        raise FileNotFoundError(
            "找不到 data/merged_output_test.csv"
        )

    raw = pd.read_csv(raw_file)

    buy_hold_value = calculate_buy_and_hold(raw)

    # ------------------------------------------------------------
    # 3. Align lengths
    # ------------------------------------------------------------

    n = min(
        len(mfn_value),
        len(buy_hold_value),
    )

    mfn_value = mfn_value.iloc[:n]
    buy_hold_value = buy_hold_value.iloc[:n]

    # ------------------------------------------------------------
    # 4. Normalize both methods to $10,000
    # ------------------------------------------------------------

    mfn_value = (
        mfn_value
        / mfn_value.iloc[0]
        * INITIAL_BALANCE
    )

    buy_hold_value = (
        buy_hold_value
        / buy_hold_value.iloc[0]
        * INITIAL_BALANCE
    )

    # ------------------------------------------------------------
    # 5. Save comparison CSV
    # ------------------------------------------------------------

    comparison = pd.DataFrame({
        "timestep": np.arange(n),
        "MFN-A2C": mfn_value.values,
        "Buy-and-Hold": buy_hold_value.values,
    })

    comparison.to_csv(
        ROOT / "mfn_vs_buyhold.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 6. Calculate metrics
    # ------------------------------------------------------------

    mfn_final = mfn_value.iloc[-1]
    bh_final = buy_hold_value.iloc[-1]

    mfn_return = (
        mfn_final / INITIAL_BALANCE - 1
    ) * 100

    bh_return = (
        bh_final / INITIAL_BALANCE - 1
    ) * 100

    mfn_peak = mfn_value.max()
    bh_peak = buy_hold_value.max()

    # ------------------------------------------------------------
    # 7. Print results
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("MFN-A2C vs BUY-AND-HOLD")
    print("=" * 60)

    print(
        f"Comparison timesteps : {n}"
    )

    print()

    print(
        f"MFN-A2C Final PV     : "
        f"{mfn_final:.2f}"
    )

    print(
        f"MFN-A2C Return       : "
        f"{mfn_return:.2f}%"
    )

    print(
        f"MFN-A2C Peak PV      : "
        f"{mfn_peak:.2f}"
    )

    print()

    print(
        f"Buy-Hold Final PV    : "
        f"{bh_final:.2f}"
    )

    print(
        f"Buy-Hold Return      : "
        f"{bh_return:.2f}%"
    )

    print(
        f"Buy-Hold Peak PV     : "
        f"{bh_peak:.2f}"
    )

    print("=" * 60)

    # ------------------------------------------------------------
    # 8. Plot
    # ------------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        mfn_value,
        label="MFN-A2C",
        linewidth=2,
    )

    plt.plot(
        buy_hold_value,
        label="Buy-and-Hold",
        linewidth=2,
    )

    plt.axhline(
        INITIAL_BALANCE,
        linestyle="--",
        linewidth=1,
        label="Initial Portfolio ($10,000)",
    )

    plt.title(
        "MFN-A2C vs Buy-and-Hold"
    )

    plt.xlabel(
        "4-hour timestep"
    )

    plt.ylabel(
        "Portfolio Value"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output = ROOT / "mfn_vs_buyhold.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(
        f"Saved comparison CSV: "
        f"{ROOT / 'mfn_vs_buyhold.csv'}"
    )

    print(
        f"Saved comparison figure: "
        f"{output}"
    )


if __name__ == "__main__":
    main()