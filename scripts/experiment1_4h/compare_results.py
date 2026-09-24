"""彙整 4H Experiment 1 六個模型與 Buy-and-Hold。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_4h as cfg

METHODS = (
    "mfn",
    "dman_attention",
    "asset_mfn",
    "self_attention",
    "a2c",
    "without_ti",
)


def main() -> None:
    paths = [
        cfg.RESULTS / f"{cfg.MODEL_NAMES[key]}_metrics.csv"
        for key in METHODS
    ] + [cfg.RESULTS / "buy_and_hold_4h_metrics.csv"]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "請先完成六個模型與 Buy-and-Hold 評估，缺少：\n"
            + "\n".join(missing)
        )
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    buy_hold = rows.loc[rows["Method"] == "buy_and_hold"]
    buy_hold_final = float(buy_hold["Final PV"].iloc[0])
    buy_hold_peak = float(buy_hold["Peak PV"].iloc[0])
    rows["Final Improve"] = rows["Final PV"] / buy_hold_final
    rows["Peak Improve"] = rows["Peak PV"] / buy_hold_peak
    columns = [
        "Method",
        "Peak PV",
        "Peak Improve",
        "Final PV",
        "Final Improve",
        "Total Return",
        "Max Drawdown",
        "Sharpe Ratio",
    ]
    output = cfg.RESULTS / f"experiment1_4h_{cfg.STEP_TAG}_comparison.csv"
    rows[columns].to_csv(output, index=False)
    display = rows[columns].copy()
    for column in ("Total Return", "Max Drawdown"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    print(display.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
