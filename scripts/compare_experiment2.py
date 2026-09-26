"""繪製 4H Experiment 2 的 PV、DSR 與 Expanding Sharpe 三張比較圖。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config_4h as cfg
from src.dsr import calculate_dsr_series


def _replace_step_tag(name: str, steps: int) -> str:
    requested = f"_{steps // 1000}k_"
    current = f"_{cfg.STEP_TAG}_"
    if current not in name:
        raise ValueError(f"檔名缺少步數標籤 {current}：{name}")
    return name.replace(current, requested, 1)


def result_files(steps: int) -> dict[str, str]:
    return {
        "Proposed Method": _replace_step_tag(
            cfg.RESULT_NAMES["dman_attention"], steps
        ),
        "A2C": _replace_step_tag(cfg.RESULT_NAMES["a2c"], steps),
        "DQN": _replace_step_tag(cfg.DQN_RESULT_NAME, steps),
        "Buy and Hold": "buy_and_hold_4h_results.csv",
    }


def load_curve(name: str, filename: str) -> pd.DataFrame:
    """由 PV 重算報酬、正式 EWMA DSR 與年化 Expanding Sharpe。"""
    path = cfg.MODEL_RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(f"{name} 缺少 4H 結果：{path}")
    frame = pd.read_csv(path)
    required = {"timestamp", "portfolio_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少欄位：{sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["portfolio_value"] = pd.to_numeric(
        frame["portfolio_value"], errors="raise"
    )

    # 與 Experiment 1 一致，不讀模型 reward 或舊 CSV 的 return/DSR。
    frame["return"] = frame["portfolio_value"].pct_change()
    valid_returns = frame["return"].dropna()
    frame["dsr"] = np.nan
    frame.loc[valid_returns.index, "dsr"] = calculate_dsr_series(
        valid_returns.to_numpy(),
        eta=cfg.DSR_ETA,
        formula=cfg.EVALUATION_DSR_FORMULA,
    )
    frame["cumulative_dsr"] = frame["dsr"].fillna(0.0).cumsum()

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


def load_aligned_curves(steps: int) -> dict[str, pd.DataFrame]:
    """讀取四種方法，並拒絕時間戳或資料長度不一致的結果。"""
    curves = {
        name: load_curve(name, filename)
        for name, filename in result_files(steps).items()
    }
    reference = next(iter(curves.values()))["timestamp"].reset_index(drop=True)
    for name, frame in curves.items():
        timestamps = frame["timestamp"].reset_index(drop=True)
        if not timestamps.equals(reference):
            raise ValueError(f"{name} 的時間戳與其他 Experiment 2 結果不一致。")
    return curves


def _save_figure(fig, output: Path) -> None:
    """先寫入同資料夾暫存檔，再重試原子替換以避開 OneDrive 短暫鎖定。"""
    temporary = output.with_name(
        f".{output.stem}.{os.getpid()}.tmp{output.suffix}"
    )
    try:
        fig.savefig(temporary, dpi=300, bbox_inches="tight")
        last_error = None
        for attempt in range(4):
            try:
                os.replace(temporary, output)
                return
            except OSError as error:
                last_error = error
                if attempt < 3:
                    time.sleep(0.25)
        raise OSError(f"無法覆寫圖檔，請關閉正在檢視的檔案：{output}") from last_error
    finally:
        temporary.unlink(missing_ok=True)


def plot_experiment2(
    curves: dict[str, pd.DataFrame],
    step_tag: str,
) -> tuple[Path, Path, Path]:
    """將三張圖分別輸出到 figures/experiment2。"""
    length = len(next(iter(curves.values())))
    x = np.arange(length)
    output_dir = cfg.FIGURES / "experiment2"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / f"experiment2_4h_{step_tag}_portfolio_value.png",
        output_dir / f"experiment2_4h_{step_tag}_differential_sharpe_ratio.png",
        output_dir / f"experiment2_4h_{step_tag}_expanding_sharpe_ratio.png",
    )
    figure_specs = (
        ("portfolio_value", "Portfolio Value", False, outputs[0]),
        ("cumulative_dsr", "Differential Sharpe Ratio", False, outputs[1]),
        (
            "expanding_sharpe_ratio",
            "Annualized Expanding Sharpe Ratio",
            True,
            outputs[2],
        ),
    )
    for column, ylabel, zero_line, output in figure_specs:
        fig, ax = plt.subplots(figsize=(8.4, 6.4))
        for name, frame in curves.items():
            ax.plot(x, frame[column], label=name, linewidth=1.7)
        if zero_line:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("iter(4hrs)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.20)
        ax.legend(title="Method", loc="upper left")
        fig.tight_layout()
        _save_figure(fig, output)
        plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="繪製 4H Experiment 2 三張比較圖。")
    parser.add_argument(
        "--steps", type=int, default=cfg.TOTAL_TIMESTEPS,
        help="比較的模型訓練步數；預設使用 config_4h.TOTAL_TIMESTEPS。",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 1000 != 0:
        parser.error("--steps 必須是正整數且可被 1000 整除。")
    step_tag = f"{args.steps // 1000}k"

    cfg.EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    curves = load_aligned_curves(args.steps)
    reference = next(iter(curves.values()))["timestamp"].reset_index(drop=True)
    comparison = pd.DataFrame({
        "timestamp": reference,
        **{
            f"{name} Portfolio Value": frame["portfolio_value"].values
            for name, frame in curves.items()
        },
        **{
            f"{name} Cumulative DSR": frame["cumulative_dsr"].values
            for name, frame in curves.items()
        },
        **{
            f"{name} Expanding Sharpe": frame["expanding_sharpe_ratio"].values
            for name, frame in curves.items()
        },
    })
    output_csv = cfg.EXPERIMENT_RESULTS / f"experiment2_4h_{step_tag}_comparison.csv"
    comparison.to_csv(output_csv, index=False, lineterminator="\n")

    baseline = float(curves["Buy and Hold"]["portfolio_value"].iloc[-1])
    rows = []
    for name, frame in curves.items():
        values = frame["portfolio_value"]
        rows.append({
            "Method": name,
            "Peak PV": values.max(),
            "Final PV": values.iloc[-1],
            "Final Improve": values.iloc[-1] / baseline,
            "Peak Cumulative DSR": frame["cumulative_dsr"].max(),
            "Final Cumulative DSR": frame["cumulative_dsr"].iloc[-1],
            "Final Expanding Sharpe": frame["expanding_sharpe_ratio"].iloc[-1],
        })
    table = pd.DataFrame(rows)
    table_output = cfg.EXPERIMENT_RESULTS / f"experiment2_4h_{step_tag}_table.csv"
    table.to_csv(table_output, index=False, lineterminator="\n")

    pv_figure, dsr_figure, sharpe_figure = plot_experiment2(curves, step_tag)
    print(table.to_string(index=False))
    print(f"Saved: {output_csv}")
    print(f"Saved: {table_output}")
    print(f"Saved: {pv_figure}")
    print(f"Saved: {dsr_figure}")
    print(f"Saved: {sharpe_figure}")


if __name__ == "__main__":
    main()
