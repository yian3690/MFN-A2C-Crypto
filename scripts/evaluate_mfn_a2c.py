"""Backtest the GitHub-style two-view MFN-A2C model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stable_baselines3 import A2C

from src.dsr import PAPER_FORMULA
from src.mfn_github_extractor import GitHubStyleTwoViewMFN
from src.evaluation_metrics import (
    add_test_timestamps,
    add_dsr_columns,
    print_allocation_summary,
    print_test_period,
    summarize_allocations,
    summarize_dsr,
    validate_test_period,
)
from src.portfolio_env_sb3 import CryptoPortfolioEnv
from src.simplex_policy import SimplexActorCriticPolicy
from src.experiment_periods import PERIODS_PER_YEAR
from src.training_diagnostics import diagnose_training_file, latest_diagnostics


DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"


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
    test_period = validate_test_period(test_raw, 20)

    env = CryptoPortfolioEnv(

        pct_csv=str(
            DATA / "pct_change_output_test.csv"
        ),

        ta_csv=str(
            DATA / "ta_test_test.csv"
        ),

        raw_csv=str(
            DATA / "merged_output_test.csv"
        ),

        n_previous_timesteps=20,

        max_episode_steps=(
            len(pd.read_csv(
                DATA / "pct_change_output_test.csv"
            )) - 20 - 1
        ),

        reward_type="dsr",

        eta=0.005,
        dsr_formula=PAPER_FORMULA,

        initial_balance=10000,

        random_start=False,
        action_mode="simplex",
    )

    model = A2C.load(

        str(MODELS / "mfn_a2c_github2_5x20_300k_paper_dsr"),

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
        20,
    )
    result = add_dsr_columns(
        result,
        eta=env.eta,
        warmup_steps=env.dsr_warmup_steps,
        formula=env.dsr_formula,
    )

    result.to_csv(
        RESULTS / "mfn_github2_5x20_300k_paper_dsr_backtest_results.csv",
        index=False,
    )

    metrics = calculate_metrics(
        result["portfolio_value"].values
    )
    metrics.update(summarize_dsr(result))
    allocation_metrics = summarize_allocations(result)
    metrics.update(allocation_metrics)

    print()
    print("=" * 60)
    print("MFN-A2C GITHUB-STYLE BACKTEST")
    print("=" * 60)
    print_test_period(test_period)

    for key, value in metrics.items():

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
    print("=" * 60)

    diagnostics_path = latest_diagnostics(
        LOGS / "training_diagnostics",
        "mfn_a2c_5x20_paper_dsr_300k_*.csv",
    )
    if diagnostics_path is not None:
        training_summary, warnings = diagnose_training_file(diagnostics_path)
        print("TRAINING DIAGNOSTICS")
        print(f"Source CSV                  : {diagnostics_path}")
        print(f"Recorded rollouts           : {int(training_summary['rollouts'])}")
        print(f"Recent reward std           : {training_summary['reward_std_recent']:.6g}")
        print(f"Recent |reward| p99         : {training_summary['reward_abs_p99_recent']:.6g}")
        print(f"Recent explained variance   : {training_summary['explained_variance_recent']:.4f}")
        print(f"Recent MFN gradient norm    : {training_summary['mfn_gradient_norm_recent']:.6g}")
        print(f"Recent allocation entropy   : {training_summary['allocation_entropy_recent']:.4f}")
        print(f"Recent equal-weight distance: {training_summary['equal_weight_distance_recent']:.4f}")
        print(f"Recent policy weight change : {training_summary['policy_weight_variation_recent']:.4f}")
        print("Possible causes:")
        for warning in warnings:
            print(f"- {warning}")
        print("=" * 60)
    else:
        print("Training diagnostics: no matching CSV found.")
        print("=" * 60)

    pd.DataFrame(
        [metrics]
    ).to_csv(
        RESULTS / "mfn_github2_5x20_300k_paper_dsr_metrics.csv",
        index=False,
    )

    env.close()


if __name__ == "__main__":
    main()
