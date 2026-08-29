from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)

INITIAL_BALANCE = 10000.0


# ============================================================
# Load curve
# ============================================================

def load_curve(filename):

    path = (
        RESULTS /
        filename
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing result file: {path}"
        )

    df = pd.read_csv(path)

    if (
        "portfolio_value"
        not in df.columns
    ):

        raise ValueError(
            f"{filename} has no "
            f"'portfolio_value' column"
        )

    values = (
        df[
            "portfolio_value"
        ]
        .astype(float)
        .reset_index(drop=True)
    )

    # Normalize all curves
    # to initial balance = 10000

    values = (
        values
        /
        values.iloc[0]
        *
        INITIAL_BALANCE
    )

    return values


# ============================================================
# Main
# ============================================================

def main():

    curves = {

        "MFN-A2C + DSR":
            load_curve(
                "exp3_mfn_dsr_results.csv"
            ),

        "MFN-A2C + PV":
            load_curve(
                "exp3_mfn_pv_results.csv"
            ),

        "A2C + DSR":
            load_curve(
                "exp3_a2c_dsr_results.csv"
            ),

        "A2C + PV":
            load_curve(
                "exp3_a2c_pv_results.csv"
            ),
    }

    # --------------------------------------------------------
    # Align curve lengths
    # --------------------------------------------------------

    n = min(
        len(v)
        for v in curves.values()
    )

    for key in curves:

        curves[key] = (
            curves[key]
            .iloc[:n]
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    comparison = (
        pd.DataFrame({

            "timestep":
                range(n),

            **{
                method:
                    values.values

                for (
                    method,
                    values
                ) in curves.items()
            }
        })
    )

    output_csv = (
        RESULTS /
        "experiment3_comparison.csv"
    )

    comparison.to_csv(
        output_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Reference baseline
    # --------------------------------------------------------

    baseline_final = (
        comparison[
            "A2C + PV"
        ].iloc[-1]
    )

    # --------------------------------------------------------
    # Print Table-IV-like results
    # --------------------------------------------------------

    print()
    print("=" * 85)

    print(
        "EXPERIMENT 3 RESULTS"
    )

    print("=" * 85)

    print(
        f"{'Method':<20}"
        f"{'Peak PV':>15}"
        f"{'Peak Improve':>15}"
        f"{'Final PV':>15}"
        f"{'Final Improve':>15}"
    )

    print("-" * 85)

    table_rows = []

    for method in [

        "MFN-A2C + DSR",
        "MFN-A2C + PV",
        "A2C + DSR",
        "A2C + PV",

    ]:

        values = (
            comparison[
                method
            ]
        )

        peak = values.max()

        final = (
            values.iloc[-1]
        )

        # Paper defines Improve
        # relative to A2C + PV

        peak_improve = (
            peak /
            comparison[
                "A2C + PV"
            ].max()
        )

        final_improve = (
            final /
            baseline_final
        )

        print(

            f"{method:<20}"

            f"{peak:>15.2f}"

            f"{peak_improve:>15.3f}"

            f"{final:>15.2f}"

            f"{final_improve:>15.3f}"
        )

        table_rows.append({

            "Method":
                method,

            "Peak PV":
                peak,

            "Peak Improve":
                peak_improve,

            "Final PV":
                final,

            "Final Improve":
                final_improve,
        })

    print("=" * 85)

    # --------------------------------------------------------
    # Save result table
    # --------------------------------------------------------

    table_df = (
        pd.DataFrame(
            table_rows
        )
    )

    table_output = (
        RESULTS /
        "experiment3_table.csv"
    )

    table_df.to_csv(
        table_output,
        index=False,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    for method in [

        "MFN-A2C + DSR",
        "MFN-A2C + PV",
        "A2C + DSR",
        "A2C + PV",

    ]:

        plt.plot(

            comparison[
                method
            ],

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
        "Results of Experiment 3"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    figure_output = (
        FIGURES /
        "experiment3_comparison.png"
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
    print(
        f"Comparison CSV : "
        f"{output_csv}"
    )

    print(
        f"Result Table   : "
        f"{table_output}"
    )

    print(
        f"Figure         : "
        f"{figure_output}"
    )


if __name__ == "__main__":
    main()