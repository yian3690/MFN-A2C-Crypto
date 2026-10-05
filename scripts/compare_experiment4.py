"""繪製 Experiment 4：Proposed、Original MFN、A2C 與 Buy-and-Hold。"""

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
from compare_experiment2 import _replace_step_tag, _save_figure, load_curve


METHOD_LABELS = {
    "proposed": "Proposed Method",
    "original_mfn": "Original MFN-A2C",
    "a2c": "A2C",
    "buy_hold": "Buy and Hold",
}


def result_files(steps: int) -> dict[str, str]:
    """依指定步數解析四種方法的正式評估結果。"""
    return {
        METHOD_LABELS["proposed"]: _replace_step_tag(
            cfg.RESULT_NAMES["dman_attention"], steps
        ),
        METHOD_LABELS["original_mfn"]: _replace_step_tag(
            cfg.RESULT_NAMES["original_mfn"], steps
        ),
        METHOD_LABELS["a2c"]: _replace_step_tag(
            cfg.RESULT_NAMES["a2c"], steps
        ),
        METHOD_LABELS["buy_hold"]: "buy_and_hold_4h_results.csv",
    }


def load_aligned_curves(steps: int) -> dict[str, pd.DataFrame]:
    """載入四條曲線，並拒絕不同測試時間軸。"""
    curves = {
        name: load_curve(name, filename)
        for name, filename in result_files(steps).items()
    }
    reference = next(iter(curves.values()))["timestamp"].reset_index(drop=True)
    for name, frame in curves.items():
        if not frame["timestamp"].reset_index(drop=True).equals(reference):
            raise ValueError(f"{name} 的時間戳與其他 Experiment 4 結果不一致。")
    return curves


def plot_experiment4(
    curves: dict[str, pd.DataFrame],
    step_tag: str,
) -> tuple[Path, Path, Path]:
    """分開輸出 PV、Cumulative DSR 與 Expanding Sharpe 圖。"""
    output_dir = cfg.FIGURES / "experiment4"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / f"experiment4_4h_{step_tag}_portfolio_value.png",
        output_dir / f"experiment4_4h_{step_tag}_differential_sharpe_ratio.png",
        output_dir / f"experiment4_4h_{step_tag}_expanding_sharpe_ratio.png",
    )
    specs = (
        ("portfolio_value", "Portfolio Value", False, outputs[0]),
        ("cumulative_dsr", "Differential Sharpe Ratio", False, outputs[1]),
        (
            "expanding_sharpe_ratio",
            "Annualized Expanding Sharpe Ratio",
            True,
            outputs[2],
        ),
    )
    x = np.arange(len(next(iter(curves.values()))))
    for column, ylabel, zero_line, output in specs:
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
    parser = argparse.ArgumentParser(
        description="繪製 4H Experiment 4 四方法比較圖。"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=cfg.TOTAL_TIMESTEPS,
        help="比較的訓練步數；預設使用 config_4h.TOTAL_TIMESTEPS。",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 1000 != 0:
        parser.error("--steps 必須是正整數且可被 1000 整除。")
    step_tag = f"{args.steps // 1000}k"

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
            f"{name} Expanding Sharpe": frame[
                "expanding_sharpe_ratio"
            ].values
            for name, frame in curves.items()
        },
    })

    cfg.EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    output_csv = (
        cfg.EXPERIMENT_RESULTS
        / f"experiment4_4h_{step_tag}_comparison.csv"
    )
    comparison.to_csv(output_csv, index=False, lineterminator="\n")

    buy_hold = curves[METHOD_LABELS["buy_hold"]]["portfolio_value"]
    rows = []
    for name, frame in curves.items():
        values = frame["portfolio_value"]
        returns = values.pct_change().dropna()
        rows.append({
            "Method": name,
            "Peak PV": values.max(),
            "Peak Improve": values.max() / buy_hold.max(),
            "Final PV": values.iloc[-1],
            "Final Improve": values.iloc[-1] / buy_hold.iloc[-1],
            "Total Return": values.iloc[-1] / values.iloc[0] - 1.0,
            "Max Drawdown": (values / values.cummax() - 1.0).min(),
            "Sharpe Ratio": (
                returns.mean()
                / returns.std()
                * np.sqrt(cfg.PERIODS_PER_YEAR)
                if returns.std() > 0
                else 0.0
            ),
            "Peak Cumulative DSR": frame["cumulative_dsr"].max(),
            "Final Cumulative DSR": frame["cumulative_dsr"].iloc[-1],
            "Final Expanding Sharpe": frame[
                "expanding_sharpe_ratio"
            ].iloc[-1],
        })
    table = pd.DataFrame(rows)
    table_output = (
        cfg.EXPERIMENT_RESULTS / f"experiment4_4h_{step_tag}_table.csv"
    )
    table.to_csv(table_output, index=False, lineterminator="\n")

    pv_figure, dsr_figure, sharpe_figure = plot_experiment4(
        curves, step_tag
    )
    print(table.to_string(index=False))
    print(f"Saved: {output_csv}")
    print(f"Saved: {table_output}")
    print(f"Saved: {pv_figure}")
    print(f"Saved: {dsr_figure}")
    print(f"Saved: {sharpe_figure}")


if __name__ == "__main__":
    main()
