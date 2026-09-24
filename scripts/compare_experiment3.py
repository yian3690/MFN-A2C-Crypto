"""繪製4H Experiment 3：A2C的DSR與絕對Portfolio Value reward消融。"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config_4h as cfg

METHODS = {
    "A2C + DSR": cfg.RESULT_NAMES["a2c"],
    "A2C + PV": cfg.A2C_PV_RESULT_NAME,
}


def _load(name: str, filename: str) -> pd.DataFrame:
    path = cfg.RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(f"{name}缺少4H結果：{path}")
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def main() -> None:
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    curves = {name: _load(name, filename) for name, filename in METHODS.items()}
    common = min(len(frame) for frame in curves.values())
    comparison = pd.DataFrame({
        "timestamp": next(iter(curves.values()))["timestamp"].iloc[:common].values,
        **{name: frame["portfolio_value"].iloc[:common].astype(float).values
           for name, frame in curves.items()},
    })
    output_csv = cfg.RESULTS / f"experiment3_4h_{cfg.STEP_TAG}_comparison.csv"
    comparison.to_csv(output_csv, index=False)

    baseline = float(comparison["A2C + PV"].iloc[-1])
    rows = []
    for name in METHODS:
        values = comparison[name]
        rows.append({"Method": name, "Peak PV": values.max(),
                     "Peak Improve": values.max() / comparison["A2C + PV"].max(),
                     "Final PV": values.iloc[-1],
                     "Final Improve": values.iloc[-1] / baseline})
    table = pd.DataFrame(rows)
    table.to_csv(cfg.RESULTS / f"experiment3_4h_{cfg.STEP_TAG}_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in METHODS:
        ax.plot(comparison["timestamp"], comparison[name], label=name, linewidth=1.8)
    ax.set_title("Experiment 3 (4H): Effect of Reward Selection")
    ax.set_xlabel("Date (4-hour bars)"); ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    output_figure = cfg.FIGURES / f"experiment3_4h_{cfg.STEP_TAG}_comparison.png"
    fig.savefig(output_figure, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(table.to_string(index=False)); print(f"Saved: {output_csv}"); print(f"Saved: {output_figure}")


if __name__ == "__main__":
    main()
