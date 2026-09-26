"""彙整4H Experiment 1，並以論文EWMA actual-change DSR輸出分開圖表。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_4h as cfg
from src.dsr import calculate_dsr_series
from src.evaluation_metrics import summarize_dsr

METHODS = (
    "dman_attention",
    "a2c",
    "without_ti",
)
METHOD_LABELS = {
    "dman_attention": "Proposed Method",
    "a2c": "A2C",
    "without_ti": "A2C w/o ti",
    "buy_and_hold": "Buy and Hold",
}


def _replace_step_tag(name: str, steps: int) -> str:
    """以指定步數讀取既有結果，不受目前訓練目標步數限制。"""
    requested_tag = f"{steps // 1000}k"
    current_token = f"_{cfg.STEP_TAG}_"
    if current_token not in name:
        raise ValueError(f"檔名缺少目前步數標籤{current_token}：{name}")
    return name.replace(current_token, f"_{requested_tag}_", 1)


def result_files(steps: int) -> dict[str, str]:
    files = {
        METHOD_LABELS[key]: _replace_step_tag(cfg.RESULT_NAMES[key], steps)
        for key in METHODS
    }
    files[METHOD_LABELS["buy_and_hold"]] = "buy_and_hold_4h_results.csv"
    return files


def metric_files(steps: int) -> dict[str, Path]:
    files = {
        key: cfg.MODEL_RESULTS
        / _replace_step_tag(f"{cfg.MODEL_NAMES[key]}_metrics.csv", steps)
        for key in METHODS
    }
    files["buy_and_hold"] = cfg.MODEL_RESULTS / "buy_and_hold_4h_metrics.csv"
    return files


def load_curve(label: str, filename: str) -> pd.DataFrame:
    """由PV報酬重算論文EWMA actual-change DSR。"""
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

    # 正式評估固定採PV_t / PV_{t-1} - 1，不讀模型reward或舊版return。
    frame["return"] = frame["portfolio_value"].pct_change()
    valid_returns = frame["return"].dropna()
    step_dsr = calculate_dsr_series(
        valid_returns.to_numpy(),
        eta=cfg.DSR_ETA,
        formula=cfg.EVALUATION_DSR_FORMULA,
    )
    frame["dsr"] = np.nan
    frame.loc[valid_returns.index, "dsr"] = step_dsr
    frame["cumulative_dsr"] = frame["dsr"].fillna(0.0).cumsum()

    # 與報表Sharpe採相同4H年化係數；最後一點應等於整段回測Sharpe。
    expanding = frame["return"].expanding(
        min_periods=cfg.RELATIVE_STRENGTH_BARS
    )
    expanding_std = expanding.std(ddof=1)
    frame["expanding_sharpe_ratio"] = (
        expanding.mean()
        / expanding_std.where(expanding_std > 0.0)
        * np.sqrt(cfg.PERIODS_PER_YEAR)
    )
    return frame


def _load_aligned_curves(steps: int) -> dict[str, pd.DataFrame]:
    """讀取四條曲線並拒絕時間戳不一致的結果。"""
    curves = {
        label: load_curve(label, name)
        for label, name in result_files(steps).items()
    }
    reference = next(iter(curves.values()))["timestamp"].reset_index(drop=True)
    for label, frame in curves.items():
        timestamps = frame["timestamp"].reset_index(drop=True)
        if not timestamps.equals(reference):
            raise ValueError(f"{label}的時間戳與其他Experiment 1結果不一致。")
    return curves


def plot_experiment1(
    curves: dict[str, pd.DataFrame],
    step_tag: str,
) -> tuple[Path, Path, Path]:
    """分別輸出PV、EWMA cumulative DSR與expanding Sharpe圖。"""
    length = len(next(iter(curves.values())))
    x = np.arange(length)

    output_dir = cfg.FIGURES / "experiment1"
    output_dir.mkdir(parents=True, exist_ok=True)
    pv_output = output_dir / f"experiment1_4h_{step_tag}_portfolio_value.png"
    dsr_output = output_dir / (
        f"experiment1_4h_{step_tag}_differential_sharpe_ratio.png"
    )
    sharpe_output = output_dir / (
        f"experiment1_4h_{step_tag}_expanding_sharpe_ratio.png"
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
        ax.plot(x, frame["cumulative_dsr"], label=label, linewidth=1.7)
    ax.set_xlabel("iter(4hrs)")
    ax.set_ylabel("Differential Sharpe Ratio")
    ax.grid(alpha=0.20)
    ax.legend(title="Method", loc="upper left")
    fig.tight_layout()
    fig.savefig(dsr_output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for label, frame in curves.items():
        ax.plot(x, frame["expanding_sharpe_ratio"], label=label, linewidth=1.7)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("iter(4hrs)")
    ax.set_ylabel("Annualized Expanding Sharpe Ratio")
    ax.grid(alpha=0.20)
    ax.legend(title="Method", loc="upper left")
    fig.tight_layout()
    fig.savefig(sharpe_output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pv_output, dsr_output, sharpe_output


def main() -> None:
    parser = argparse.ArgumentParser(description="繪製4H Experiment 1比較圖。")
    parser.add_argument(
        "--steps",
        type=int,
        default=cfg.TOTAL_TIMESTEPS,
        help="比較的模型訓練步數；預設使用config_4h.TOTAL_TIMESTEPS。",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 1000 != 0:
        parser.error("--steps必須是正整數且可被1000整除。")
    step_tag = f"{args.steps // 1000}k"

    files = metric_files(args.steps)
    paths = list(files.values())
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "請先完成三個模型與Buy-and-Hold評估，缺少：\n"
            + "\n".join(missing)
        )

    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    curves = _load_aligned_curves(args.steps)

    # 不採用舊metrics中的DSR；四種方法都由相同PV公式重新計算。
    for method, label in METHOD_LABELS.items():
        dsr_metrics = summarize_dsr(curves[label])
        selection = rows["Method"] == method
        for column, value in dsr_metrics.items():
            rows.loc[selection, column] = value

    buy_hold = rows.loc[rows["Method"] == "buy_and_hold"]
    buy_hold_final = float(buy_hold["Final PV"].iloc[0])
    buy_hold_peak = float(buy_hold["Peak PV"].iloc[0])
    rows["DSR Formula"] = cfg.EVALUATION_DSR_FORMULA
    rows["DSR Eta"] = cfg.DSR_ETA
    rows["Final Improve"] = rows["Final PV"] / buy_hold_final
    rows["Peak Improve"] = rows["Peak PV"] / buy_hold_peak
    columns = [
        "Method",
        "DSR Formula",
        "DSR Eta",
        "Peak PV",
        "Peak Improve",
        "Final PV",
        "Final Improve",
        "Total Return",
        "Max Drawdown",
        "Sharpe Ratio",
        "Peak Cumulative DSR",
        "Final Cumulative DSR",
    ]
    cfg.EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    output = cfg.EXPERIMENT_RESULTS / f"experiment1_{step_tag}_comparison.csv"
    rows[columns].to_csv(output, index=False, lineterminator="\n")

    display = rows[columns].copy()
    for column in ("Total Return", "Max Drawdown"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    print(display.to_string(index=False))
    print(f"\nSaved: {output}")

    pv_figure, dsr_figure, sharpe_figure = plot_experiment1(curves, step_tag)
    print(f"Saved: {pv_figure}")
    print(f"Saved: {dsr_figure}")
    print(f"Saved: {sharpe_figure}")


if __name__ == "__main__":
    main()