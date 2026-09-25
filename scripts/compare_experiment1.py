"""彙整4H Experiment 1，並輸出論文相容的PV與DSR圖。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_4h as cfg
from src.dsr import LEGACY_EXPANDING_FORMULA, calculate_dsr_series

METHODS = (
    "dman_attention",
    "a2c",
    "without_ti",
)

CURVES = {
    "Proposed Method": cfg.RESULT_NAMES["dman_attention"],
    "A2C": cfg.RESULT_NAMES["a2c"],
    "A2C w/o ti": cfg.RESULT_NAMES["without_ti"],
    "Buy and Hold": "buy_and_hold_4h_results.csv",
}


def load_curve(label: str, filename: str) -> pd.DataFrame:
    """讀取逐步回測，並只為論文圖表重算學長版累積DSR。"""
    path = cfg.MODEL_RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(f"{label}缺少逐步結果：{path}")
    frame = pd.read_csv(path)
    required = {"timestamp", "portfolio_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}缺少欄位：{sorted(missing)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["portfolio_value"] = pd.to_numeric(
        frame["portfolio_value"], errors="raise"
    )
    if "return" not in frame:
        frame["return"] = frame["portfolio_value"].pct_change()

    # 學長論文圖表使用 expanding moments，且每期 Dt 乘上 eta。
    # 這個欄位只供繪圖，不會改變訓練 reward、模型或正式評估CSV。
    valid_returns = pd.to_numeric(frame["return"], errors="coerce").dropna()
    step_dsr = calculate_dsr_series(
        valid_returns.to_numpy(),
        eta=cfg.DSR_ETA,
        formula=LEGACY_EXPANDING_FORMULA,
    )
    frame["thesis_cumulative_dsr"] = 0.0
    frame.loc[valid_returns.index, "thesis_cumulative_dsr"] = step_dsr
    frame["thesis_cumulative_dsr"] = frame["thesis_cumulative_dsr"].cumsum()
    return frame


def _load_aligned_curves() -> dict[str, pd.DataFrame]:
    """讀取四條曲線並拒絕時間戳不一致的結果。"""
    curves = {label: load_curve(label, name) for label, name in CURVES.items()}
    reference = next(iter(curves.values()))["timestamp"].reset_index(drop=True)
    for label, frame in curves.items():
        timestamps = frame["timestamp"].reset_index(drop=True)
        if not timestamps.equals(reference):
            raise ValueError(f"{label}的時間戳與其他Experiment 1結果不一致。")
    return curves


def plot_experiment1() -> tuple[Path, Path]:
    """分別輸出Portfolio Value與論文相容DSR圖。"""
    curves = _load_aligned_curves()
    length = len(next(iter(curves.values())))
    x = np.arange(length)

    output_dir = cfg.FIGURES / "experiment1"
    output_dir.mkdir(parents=True, exist_ok=True)
    pv_output = output_dir / (
        f"experiment1_4h_{cfg.STEP_TAG}_portfolio_value.png"
    )
    dsr_output = output_dir / (
        f"experiment1_4h_{cfg.STEP_TAG}_differential_sharpe_ratio.png"
    )

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for label, frame in curves.items():
        ax.plot(x, frame["portfolio_value"], label=label, linewidth=1.7)
    ax.set_xlabel("iter(4hrs)")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.20)
    ax.legend(title="Method", loc="upper left")
    fig.tight_layout()
    fig.savefig(pv_output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for label, frame in curves.items():
        ax.plot(x, frame["thesis_cumulative_dsr"], label=label, linewidth=1.7)
    ax.set_xlabel("iter(4hrs)")
    ax.set_ylabel("Differential Sharpe Ratio")
    ax.grid(alpha=0.20)
    ax.legend(title="Method", loc="upper left")
    fig.tight_layout()
    fig.savefig(dsr_output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pv_output, dsr_output


def main() -> None:
    paths = [
        cfg.MODEL_RESULTS / f"{cfg.MODEL_NAMES[key]}_metrics.csv"
        for key in METHODS
    ] + [cfg.MODEL_RESULTS / "buy_and_hold_4h_metrics.csv"]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "請先完成三個模型與 Buy-and-Hold 評估，缺少：\n"
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
    cfg.EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    output = cfg.EXPERIMENT_RESULTS / f"experiment1_{cfg.STEP_TAG}_comparison.csv"
    rows[columns].to_csv(output, index=False)

    display = rows[columns].copy()
    for column in ("Total Return", "Max Drawdown"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    print(display.to_string(index=False))
    print(f"\nSaved: {output}")

    pv_figure, dsr_figure = plot_experiment1()
    print(f"Saved: {pv_figure}")
    print(f"Saved: {dsr_figure}")


if __name__ == "__main__":
    main()