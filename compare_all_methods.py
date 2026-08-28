from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

INITIAL_BALANCE = 10000.0


def load_curve(filename):

    df = pd.read_csv(
        ROOT / filename
    )

    values = (
        df["portfolio_value"]
        .astype(float)
        .reset_index(drop=True)
    )

    values = (
        values /
        values.iloc[0] *
        INITIAL_BALANCE
    )

    return values


def calculate_return(values):

    return (
        values.iloc[-1] /
        values.iloc[0] -
        1
    ) * 100


def main():

    proposed = load_curve(
        "formal_backtest_results.csv"
    )

    a2c = load_curve(
        "a2c_baseline_results.csv"
    )

    a2c_without_ti = load_curve(
        "a2c_without_ti_results.csv"
    )

    buy_hold = load_curve(
        "buy_hold_results.csv"
    )

    n = min(
        len(proposed),
        len(a2c),
        len(a2c_without_ti),
        len(buy_hold),
    )

    proposed = proposed.iloc[:n]
    a2c = a2c.iloc[:n]
    a2c_without_ti = (
        a2c_without_ti.iloc[:n]
    )
    buy_hold = buy_hold.iloc[:n]

    comparison = pd.DataFrame({

        "timestep": range(n),

        "Proposed Method":
            proposed.values,

        "A2C":
            a2c.values,

        "A2C w/o TI":
            a2c_without_ti.values,

        "Buy and Hold":
            buy_hold.values,
    })

    comparison.to_csv(
        ROOT /
        "experiment1_comparison.csv",

        index=False,
    )

    print()
    print("=" * 75)
    print("EXPERIMENT 1 COMPARISON")
    print("=" * 75)

    for method in [
        "Proposed Method",
        "A2C",
        "A2C w/o TI",
        "Buy and Hold",
    ]:

        values = comparison[
            method
        ]

        final = values.iloc[-1]

        peak = values.max()

        total_return = (
            final /
            INITIAL_BALANCE -
            1
        ) * 100

        improve = (
            final /
            comparison[
                "Buy and Hold"
            ].iloc[-1]
        )

        print(
            f"{method:<20}"
            f"Final PV = {final:>10.2f}   "
            f"Peak PV = {peak:>10.2f}   "
            f"Return = {total_return:>7.2f}%   "
            f"Improve = {improve:>6.3f}"
        )

    print("=" * 75)

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        comparison["Proposed Method"],
        label="Proposed Method",
        linewidth=2,
    )

    plt.plot(
        comparison["A2C"],
        label="A2C",
        linewidth=2,
    )

    plt.plot(
        comparison["A2C w/o TI"],
        label="A2C w/o TI",
        linewidth=2,
    )

    plt.plot(
        comparison["Buy and Hold"],
        label="Buy and Hold",
        linewidth=2,
    )

    plt.xlabel(
        "4-hour timestep"
    )

    plt.ylabel(
        "Portfolio Value"
    )

    plt.title(
        "Results of Experiment 1"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output = (
        ROOT /
        "experiment1_comparison.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(
        f"Saved: {output}"
    )

    print(
        f"Saved: {ROOT / 'experiment1_comparison.csv'}"
    )


if __name__ == "__main__":
    main()