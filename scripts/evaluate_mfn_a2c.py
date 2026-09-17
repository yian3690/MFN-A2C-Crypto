"""Backtest the GitHub-style two-view MFN-A2C model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stable_baselines3 import A2C

from src.mfn_github_extractor import GitHubStyleTwoViewMFN
from src.experiment_config import (
    A2C_ACTION_MODE,
    DATA,
    LOGS,
    MFN_MODEL_NAME,
    MFN_RESULT_NAME,
    MODELS,
    RESULTS,
    RUN_TAG,
    make_portfolio_env,
)
from src.evaluation_metrics import (
    add_test_timestamps,
    add_dsr_columns,
    add_strength_alignment_columns,
    print_allocation_summary,
    print_asset_contribution_summary,
    print_strength_alignment_summary,
    print_test_period,
    summarize_allocations,
    summarize_asset_contributions,
    summarize_dsr,
    summarize_strength_alignment,
    validate_test_period,
)
from src.experiment_periods import LOOKBACK, PERIODS_PER_YEAR
from src.training_diagnostics import (
    diagnose_training_file,
    latest_diagnostics,
    print_training_diagnostics_summary,
)


def calculate_metrics(values):

    """Compute return, peak value, drawdown, and annualized Sharpe ratio from a value curve."""
    values = np.asarray(values, dtype=np.float64)

    initial = values[0]
    final = values[-1]

    total_return = (
        final / initial - 1.0
    )

    running_max = np.maximum.accumulate(values)

    drawdown = (
        values / running_max - 1.0
    )

    max_drawdown = drawdown.min()

    returns = values[1:] / values[:-1] - 1.0

    if len(returns) > 1 and returns.std() > 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(PERIODS_PER_YEAR)
        )

    else:
        sharpe = 0.0

    return {
        "Initial PV": initial,
        "Final PV": final,
        "Total Return": total_return * 100,
        "Peak PV": values.max(),
        "Max Drawdown": max_drawdown * 100,
        "Sharpe Ratio": sharpe,
    }


def main():
    """主程式入口：依序執行此腳本定義的完整流程。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    test_raw = pd.read_csv(DATA / "merged_output_test.csv")
    test_period = validate_test_period(test_raw, LOOKBACK)
    env = make_portfolio_env("test", action_mode=A2C_ACTION_MODE)

    model = A2C.load(

        str(MODELS / MFN_MODEL_NAME),

        env=env,

        # Evaluation 不需要 GPU
        device="cpu",
    )

    obs, info = env.reset()

    done = False

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

    result = add_test_timestamps(
        env.get_results(),
        test_raw,
        LOOKBACK,
    )
    result = add_dsr_columns(
        result,
        eta=env.eta,
        warmup_steps=env.dsr_warmup_steps,
        formula=env.dsr_formula,
    )
    result = add_strength_alignment_columns(result)

    result.to_csv(
        RESULTS / MFN_RESULT_NAME,
        index=False,
    )

    metrics = calculate_metrics(
        result["portfolio_value"].values
    )
    metrics.update(summarize_dsr(result))
    allocation_metrics = summarize_allocations(result)
    metrics.update(allocation_metrics)
    contribution_metrics = summarize_asset_contributions(result)
    metrics.update(contribution_metrics)
    strength_metrics = summarize_strength_alignment(result)
    metrics.update(strength_metrics)

    print()
    print("=" * 60)
    print("MFN-A2C GITHUB-STYLE BACKTEST")
    print("=" * 60)
    print_test_period(test_period)

    for key, value in metrics.items():

        if key in contribution_metrics or key in strength_metrics:
            continue

        if "Return" in key or "Drawdown" in key:

            print(
                f"{key:<20}: {value:.2f}%"
            )

        else:

            print(
                f"{key:<20}: {value:.4f}"
            )

    print("=" * 60)
    print_allocation_summary(allocation_metrics)
    print_asset_contribution_summary(contribution_metrics)
    print_strength_alignment_summary(strength_metrics)
    print("=" * 60)

    diagnostics_path = latest_diagnostics(
        LOGS / "training_diagnostics",
        f"mfn_a2c_{RUN_TAG}_*.csv",
    )
    if diagnostics_path is not None:
        training_summary, warnings = diagnose_training_file(diagnostics_path)
        print_training_diagnostics_summary(
            diagnostics_path,
            training_summary,
            warnings,
        )
        print("=" * 60)
    else:
        print("Training diagnostics: no matching CSV found.")
        print("=" * 60)

    pd.DataFrame(
        [metrics]
    ).to_csv(
        RESULTS / MFN_RESULT_NAME.replace("_results.csv", "_metrics.csv"),
        index=False,
    )

    env.close()


if __name__ == "__main__":
    main()
