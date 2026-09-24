"""繪製4H Experiment 2：Temporal Attention、A2C、DQN與Buy-and-Hold。"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config_4h as cfg

<<<<<<< Updated upstream
from src.evaluation_metrics import validate_saved_result_period

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)

INITIAL_BALANCE = 10000.0
=======
METHODS = {
    "Proposed DMAN-Temporal-Attention A2C": cfg.RESULT_NAMES["dman_attention"],
    "A2C": cfg.RESULT_NAMES["a2c"],
    "DQN": cfg.DQN_RESULT_NAME,
    "Buy-and-Hold": "buy_and_hold_4h_results.csv",
}
>>>>>>> Stashed changes


def _load(name: str, filename: str) -> pd.DataFrame:
    path = cfg.RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(f"{name}缺少4H結果：{path}")
    frame = pd.read_csv(path)
    if "portfolio_value" not in frame:
        raise ValueError(f"{path}沒有portfolio_value欄位。")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


<<<<<<< Updated upstream
def load_cumulative_dsr(filename):
    """Load explicitly named cumulative DSR metrics from a result CSV."""
    frame = pd.read_csv(RESULTS / filename)
    validate_saved_result_period(frame)
    if "cumulative_dsr" not in frame.columns:
        raise ValueError(
            f"{filename} has no cumulative_dsr column; run its evaluator first."
        )
    cumulative = pd.to_numeric(frame["cumulative_dsr"], errors="coerce").dropna()
    return float(cumulative.max()), float(cumulative.iloc[-1])

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

=======
def main() -> None:
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    curves = {name: _load(name, filename) for name, filename in METHODS.items()}
    common = min(len(frame) for frame in curves.values())
>>>>>>> Stashed changes
    comparison = pd.DataFrame({
        "timestamp": next(iter(curves.values()))["timestamp"].iloc[:common].values,
        **{name: frame["portfolio_value"].iloc[:common].astype(float).values
           for name, frame in curves.items()},
    })
    output_csv = cfg.RESULTS / f"experiment2_4h_{cfg.STEP_TAG}_comparison.csv"
    comparison.to_csv(output_csv, index=False)

    baseline = float(comparison["Buy-and-Hold"].iloc[-1])
    rows = []
    for name in METHODS:
        values = comparison[name]
        rows.append({"Method": name, "Peak PV": values.max(),
                     "Final PV": values.iloc[-1],
                     "Final Improve": values.iloc[-1] / baseline})
    table = pd.DataFrame(rows)
    table.to_csv(cfg.RESULTS / f"experiment2_4h_{cfg.STEP_TAG}_table.csv", index=False)

<<<<<<< Updated upstream
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

    result_files = {
        "Proposed Method": "formal_backtest_results.csv",
        "A2C": "a2c_baseline_results.csv",
        "DQN": "dqn_baseline_results.csv",
        "Buy and Hold": "buy_hold_results.csv",
    }
    dsr_rows = []
    print()
    print("CUMULATIVE DSR")
    print(f"{'Method':<20}{'Peak Cumulative DSR':>24}{'Final Cumulative DSR':>25}")
    print("-" * 69)
    for method, filename in result_files.items():
        peak_dsr, final_dsr = load_cumulative_dsr(filename)
        print(f"{method:<20}{peak_dsr:>24.4f}{final_dsr:>25.4f}")
        dsr_rows.append({
            "Method": method,
            "Peak Cumulative DSR": peak_dsr,
            "Final Cumulative DSR": final_dsr,
        })

    pd.DataFrame(dsr_rows).to_csv(
        RESULTS / "experiment2_dsr_metrics.csv",
        index=False,
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
        "2-hour timestep"
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
=======
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in METHODS:
        ax.plot(comparison["timestamp"], comparison[name], label=name, linewidth=1.8)
    ax.set_title("Experiment 2 (4H): Effect of Temporal Feature Extraction")
    ax.set_xlabel("Date (4-hour bars)"); ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    output_figure = cfg.FIGURES / f"experiment2_4h_{cfg.STEP_TAG}_comparison.png"
    fig.savefig(output_figure, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(table.to_string(index=False)); print(f"Saved: {output_csv}"); print(f"Saved: {output_figure}")
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()
