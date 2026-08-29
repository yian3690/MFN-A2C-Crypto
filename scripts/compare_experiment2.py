"""Create the Experiment 2 MFN-A2C, A2C, DQN, and buy-and-hold comparison."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)

INITIAL_BALANCE = 10000.0


def load_curve(filename):

    """Load and normalize one Experiment 2 portfolio-value result file."""
    df = pd.read_csv(
        RESULTS / filename
    )

    values = (
        df["portfolio_value"]
        .astype(float)
        .reset_index(drop=True)
    )

    return (
        values
        / values.iloc[0]
        * INITIAL_BALANCE
    )


def main():

    """主程式入口：依序執行此腳本定義的完整流程。"""
    proposed = load_curve(
        "formal_backtest_results.csv"
    )

    a2c = load_curve(
        "a2c_baseline_results.csv"
    )

    dqn = load_curve(
        "dqn_baseline_results.csv"
    )

    buy_hold = load_curve(
        "buy_hold_results.csv"
    )

    n = min(
        len(proposed),
        len(a2c),
        len(dqn),
        len(buy_hold),
    )

    comparison = pd.DataFrame({

        "timestep": range(n),

        "Proposed Method":
            proposed.iloc[:n].values,

        "A2C":
            a2c.iloc[:n].values,

        "DQN":
            dqn.iloc[:n].values,

        "Buy and Hold":
            buy_hold.iloc[:n].values,
    })

    comparison.to_csv(
        RESULTS /
        "experiment2_comparison.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("EXPERIMENT 2")
    print("=" * 80)

    bh_final = comparison[
        "Buy and Hold"
    ].iloc[-1]

    for method in [
        "Proposed Method",
        "A2C",
        "DQN",
        "Buy and Hold",
    ]:

        values = comparison[method]

        peak = values.max()
        final = values.iloc[-1]

        improve = (
            final / bh_final
        )

        print(
            f"{method:<20}"
            f"Peak={peak:>10.2f}  "
            f"Final={final:>10.2f}  "
            f"Improve={improve:.3f}"
        )

    plt.figure(
        figsize=(12, 6)
    )

    for method in [
        "Proposed Method",
        "A2C",
        "DQN",
        "Buy and Hold",
    ]:

        plt.plot(
            comparison[method],
            label=method,
            linewidth=2,
        )

    plt.xlabel(
        "4-hour timestep"
    )

    plt.ylabel(
        "Portfolio Value"
    )

    plt.title(
        "Results of Experiment 2"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output = (
        FIGURES /
        "experiment2_comparison.png"
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


if __name__ == "__main__":
    main()