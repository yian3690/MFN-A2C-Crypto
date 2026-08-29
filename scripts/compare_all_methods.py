from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

INITIAL_BALANCE = 10000.0


# ============================================================
# Load portfolio curve
# ============================================================

def load_curve(filename):

    filepath = RESULTS / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"找不到結果檔案：{filepath}"
        )

    df = pd.read_csv(filepath)

    if "portfolio_value" not in df.columns:
        raise ValueError(
            f"{filename} 缺少 portfolio_value 欄位"
        )

    values = (
        df["portfolio_value"]
        .astype(float)
        .reset_index(drop=True)
    )

    # Normalize to the same initial portfolio value
    values = (
        values
        / values.iloc[0]
        * INITIAL_BALANCE
    )

    return values


# ============================================================
# Calculate return
# ============================================================

def calculate_return(values):

    return (
        values.iloc[-1]
        / values.iloc[0]
        - 1
    ) * 100


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 75)
    print("Loading experiment results...")
    print("=" * 75)

    # --------------------------------------------------------
    # Load four methods
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Align lengths
    # --------------------------------------------------------

    n = min(
        len(proposed),
        len(a2c),
        len(a2c_without_ti),
        len(buy_hold),
    )

    proposed = proposed.iloc[:n]
    a2c = a2c.iloc[:n]
    a2c_without_ti = a2c_without_ti.iloc[:n]
    buy_hold = buy_hold.iloc[:n]

    # --------------------------------------------------------
    # Create comparison dataframe
    # --------------------------------------------------------

    comparison = pd.DataFrame({

        "timestep":
            range(n),

        "Proposed Method":
            proposed.values,

        "A2C":
            a2c.values,

        "A2C w/o TI":
            a2c_without_ti.values,

        "Buy and Hold":
            buy_hold.values,
    })

    # --------------------------------------------------------
    # Save comparison CSV
    # --------------------------------------------------------

    csv_output = (
        RESULTS /
        "experiment1_comparison.csv"
    )

    comparison.to_csv(
        csv_output,
        index=False,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("EXPERIMENT 1 COMPARISON")
    print("=" * 90)

    print(
        f"{'Method':<20}"
        f"{'Final PV':>15}"
        f"{'Peak PV':>15}"
        f"{'Return':>12}"
        f"{'Improve':>12}"
    )

    print("-" * 90)

    methods = [
        "Proposed Method",
        "A2C",
        "A2C w/o TI",
        "Buy and Hold",
    ]

    for method in methods:

        values = comparison[method]

        final = values.iloc[-1]

        peak = values.max()

        total_return = (
            final
            / INITIAL_BALANCE
            - 1
        ) * 100

        buy_hold_final = comparison[
            "Buy and Hold"
        ].iloc[-1]

        improve = (
            final
            / buy_hold_final
        )

        print(
            f"{method:<20}"
            f"{final:>15.2f}"
            f"{peak:>15.2f}"
            f"{total_return:>11.2f}%"
            f"{improve:>12.3f}"
        )

    print("=" * 90)

    print()
    print(
        f"Comparison timesteps: {n}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Figure formatting
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_output = (
        FIGURES /
        "experiment1_comparison.png"
    )

    plt.savefig(
        figure_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("Experiment 1 comparison completed.")
    print("=" * 75)

    print(
        f"CSV    : {csv_output}"
    )

    print(
        f"Figure : {figure_output}"
    )


if __name__ == "__main__":
    main()