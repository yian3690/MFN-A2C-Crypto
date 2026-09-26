"""繪製4H Experiment 3：Proposed/A2C在DSR與Return reward下的比較。"""

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

METHOD_LABELS = {
    "proposed_dsr": "Proposed Method_DSR",
    "proposed_return": "Proposed Method_Return",
    "a2c_dsr": "A2C_DSR",
    "a2c_return": "A2C_Return",
}


def _replace_step_tag(name: str, steps: int) -> str:
    current = f"_{cfg.STEP_TAG}_"
    requested = f"_{steps // 1000}k_"
    if current not in name:
        raise ValueError(f"檔名缺少目前步數標籤{current}：{name}")
    return name.replace(current, requested, 1)


def result_files(steps: int) -> dict[str, str]:
    return {
        METHOD_LABELS["proposed_dsr"]: _replace_step_tag(
            cfg.RESULT_NAMES["dman_attention"], steps
        ),
        METHOD_LABELS["proposed_return"]: _replace_step_tag(
            cfg.PROPOSED_RETURN_RESULT_NAME, steps
        ),
        METHOD_LABELS["a2c_dsr"]: _replace_step_tag(
            cfg.RESULT_NAMES["a2c"], steps
        ),
        METHOD_LABELS["a2c_return"]: _replace_step_tag(
            cfg.A2C_RETURN_RESULT_NAME, steps
        ),
    }


def load_aligned_curves(steps: int) -> dict[str, pd.DataFrame]:
    curves = {}
    missing = []
    for label, filename in result_files(steps).items():
        path = cfg.MODEL_RESULTS / filename
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        required = {"timestamp", "portfolio_value"}
        absent = required.difference(frame.columns)
        if absent:
            raise ValueError(f"{path}缺少欄位：{sorted(absent)}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["portfolio_value"] = pd.to_numeric(
            frame["portfolio_value"], errors="raise"
        )
        frame["return"] = frame["portfolio_value"].pct_change()
        expanding = frame["return"].expanding(
            min_periods=cfg.RELATIVE_STRENGTH_BARS
        )
        expanding_std = expanding.std(ddof=1)
        frame["expanding_sharpe_ratio"] = (
            expanding.mean()
            / expanding_std.where(expanding_std > 0.0)
            * np.sqrt(cfg.PERIODS_PER_YEAR)
        )
        curves[label] = frame
    if missing:
        raise FileNotFoundError(
            "請先完成Experiment 3四個模型的評估，缺少：\n"
            + "\n".join(missing)
        )

    reference = next(iter(curves.values()))["timestamp"].reset_index(drop=True)
    for label, frame in curves.items():
        if not frame["timestamp"].reset_index(drop=True).equals(reference):
            raise ValueError(f"{label}的時間戳與其他Experiment 3結果不一致。")
    return curves


def _save_figure(fig, output: Path) -> None:
    """使用同資料夾暫存檔，避開OneDrive短暫鎖定正式PNG。"""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="繪製4H Experiment 3四線比較圖。")
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

    curves = load_aligned_curves(args.steps)
    reference = next(iter(curves.values()))["timestamp"].reset_index(drop=True)
    comparison = pd.DataFrame({
        "timestamp": reference,
        **{
            label: frame["portfolio_value"].values
            for label, frame in curves.items()
        },
        **{
            f"{label} Expanding Sharpe": frame[
                "expanding_sharpe_ratio"
            ].values
            for label, frame in curves.items()
        },
    })
    cfg.EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    output_csv = (
        cfg.EXPERIMENT_RESULTS / f"experiment3_4h_{step_tag}_comparison.csv"
    )
    comparison.to_csv(output_csv, index=False, lineterminator="\n")

    baseline = float(comparison[METHOD_LABELS["a2c_return"]].iloc[-1])
    baseline_peak = float(comparison[METHOD_LABELS["a2c_return"]].max())
    rows = []
    for label in METHOD_LABELS.values():
        values = comparison[label]
        returns = values.pct_change().dropna()
        rows.append({
            "Method": label,
            "Peak PV": values.max(),
            "Peak Improve": values.max() / baseline_peak,
            "Final PV": values.iloc[-1],
            "Final Improve": values.iloc[-1] / baseline,
            "Total Return": values.iloc[-1] / values.iloc[0] - 1.0,
            "Max Drawdown": (values / values.cummax() - 1.0).min(),
            "Sharpe Ratio": (
                returns.mean() / returns.std() * np.sqrt(cfg.PERIODS_PER_YEAR)
                if returns.std() > 0
                else 0.0
            ),
        })
    table = pd.DataFrame(rows)
    table_output = cfg.EXPERIMENT_RESULTS / f"experiment3_4h_{step_tag}_table.csv"
    table.to_csv(table_output, index=False, lineterminator="\n")

    output_dir = cfg.FIGURES / "experiment3"
    output_dir.mkdir(parents=True, exist_ok=True)
    pv_figure = output_dir / f"experiment3_4h_{step_tag}_portfolio_value.png"
    sharpe_figure = (
        output_dir / f"experiment3_4h_{step_tag}_expanding_sharpe_ratio.png"
    )
    x = np.arange(len(comparison))
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for label in METHOD_LABELS.values():
        ax.plot(x, comparison[label], label=label, linewidth=1.7)
    ax.set_xlabel("iter(4hrs)")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.20)
    ax.legend(title="Method", loc="upper left")
    fig.tight_layout()
    _save_figure(fig, pv_figure)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for label, frame in curves.items():
        ax.plot(
            x,
            frame["expanding_sharpe_ratio"],
            label=label,
            linewidth=1.7,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("iter(4hrs)")
    ax.set_ylabel("Annualized Expanding Sharpe Ratio")
    ax.grid(alpha=0.20)
    ax.legend(title="Method", loc="upper left")
    fig.tight_layout()
    _save_figure(fig, sharpe_figure)
    plt.close(fig)

    print(table.to_string(index=False))
    print(f"Saved: {output_csv}")
    print(f"Saved: {table_output}")
    print(f"Saved: {pv_figure}")
    print(f"Saved: {sharpe_figure}")


if __name__ == "__main__":
    main()
