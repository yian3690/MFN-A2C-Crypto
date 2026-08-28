from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

INITIAL_BALANCE = 10000.0


def load_curve(
    filename,
    column="portfolio_value"
):

    df = pd.read_csv(
        ROOT / filename
    )

    values = (
        df[column]
        .astype(float)
        .reset_index(drop=True)
    )

    # Normalize all methods to exactly $10,000.
    values = (
        values /
        values.iloc[0] *
        INITIAL_BALANCE
    )

    return values


def main():

    mfn = load_curve(
        "formal_backtest_results.csv"
    )

    a2c = load_curve(
        "a2c_baseline_results.csv"
    )

    buy_hold = load_curve(
        "buy_hold_results.csv"
    )

    n = min(
        len(mfn),
        len(a2c),
        len(buy_hold),
    )

    mfn = mfn.iloc[:n]
    a2c = a2c.iloc[:n]
    buy_hold = buy_hold.iloc[:n]

    comparison = pd.DataFrame({

        "timestep":
            range(n),

        "MFN-A2C":
            mfn.values,

        "A2C":
            a2c.values,

        "Buy-and-Hold":
            buy_hold.values,
    })

    comparison.to_csv(
        ROOT /
        "all_methods_comparison.csv",

        index=False,
    )

    print()
    print("=" * 70)
    print("ALL METHODS COMPARISON")
    print("=" * 70)

    for name in [
        "MFN-A2C",
        "A2C",
        "Buy-and-Hold",
    ]:

        final = comparison[
            name
        ].iloc[-1]

        total_return = (
            final /
            INITIAL_BALANCE -
            1
        ) * 100

        peak = comparison[
            name
        ].max()

        print(
            f"{name:<18}"
            f"Final PV = {final:>10.2f}   "
            f"Return = {total_return:>7.2f}%   "
            f"Peak = {peak:>10.2f}"
        )

    print("=" * 70)

    # Plot
    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        comparison["MFN-A2C"],
        label="MFN-A2C",
        linewidth=2,
    )

    plt.plot(
        comparison["A2C"],
        label="A2C",
        linewidth=2,
    )

    plt.plot(
        comparison["Buy-and-Hold"],
        label="Buy-and-Hold",
        linewidth=2,
    )

    plt.axhline(
        INITIAL_BALANCE,
        linestyle="--",
        linewidth=1,
        label="Initial PV",
    )

    plt.xlabel(
        "4-hour timestep"
    )

    plt.ylabel(
        "Portfolio Value"
    )

    plt.title(
        "Cryptocurrency Portfolio Performance Comparison"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output = (
        ROOT /
        "all_methods_comparison.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(
        "Saved:",
        output
    )


if __name__ == "__main__":
    main()