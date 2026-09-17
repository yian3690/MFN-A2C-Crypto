"""Backtest the standard A2C baseline on the held-out test period."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from stable_baselines3 import A2C

from src.experiment_config import (
    A2C_ACTION_MODE,
    A2C_MODEL_NAME,
    A2C_RESULT_NAME,
    DATA,
    LOGS,
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


def main():
    """主程式入口：評估完整Train單階段流程產生的A2C模型。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    test_raw = pd.read_csv(DATA / "merged_output_test.csv")
    test_period = validate_test_period(test_raw, LOOKBACK)
    env = make_portfolio_env("test", action_mode=A2C_ACTION_MODE)
    model = A2C.load(
        str(MODELS / A2C_MODEL_NAME),
        env=env,
        device="cpu",
    )

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    result = add_test_timestamps(env.get_results(), test_raw, LOOKBACK)
    result = add_dsr_columns(
        result,
        eta=env.eta,
        warmup_steps=env.dsr_warmup_steps,
        formula=env.dsr_formula,
    )
    result = add_strength_alignment_columns(result)
    dsr_metrics = summarize_dsr(result)
    allocation_metrics = summarize_allocations(result)
    contribution_metrics = summarize_asset_contributions(result)
    strength_metrics = summarize_strength_alignment(result)
    output = RESULTS / A2C_RESULT_NAME
    result.to_csv(output, index=False)

    values = result["portfolio_value"].astype(float)
    initial = values.iloc[0]
    final = values.iloc[-1]
    total_return = (final / initial - 1) * 100
    peak = values.max()
    max_drawdown = (values / values.cummax() - 1).min() * 100
    returns = values.pct_change().dropna()
    sharpe = 0.0
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * PERIODS_PER_YEAR**0.5

    print()
    print("=" * 60)
    print("A2C BASELINE BACKTEST")
    print("=" * 60)
    print_test_period(test_period)
    print(f"Initial PV      : {initial:.2f}")
    print(f"Final PV        : {final:.2f}")
    print(f"Total Return    : {total_return:.2f}%")
    print(f"Peak PV         : {peak:.2f}")
    print(f"Max Drawdown    : {max_drawdown:.2f}%")
    print(f"Sharpe Ratio    : {sharpe:.4f}")
    print(
        "Peak Cumulative DSR        : "
        f"{dsr_metrics['Peak Cumulative DSR']:.4f}"
    )
    print(
        "Final Cumulative DSR       : "
        f"{dsr_metrics['Final Cumulative DSR']:.4f}"
    )
    print("=" * 60)
    print_allocation_summary(allocation_metrics)
    print_asset_contribution_summary(contribution_metrics)
    print_strength_alignment_summary(strength_metrics)
    print("=" * 60)
    diagnostics_path = latest_diagnostics(
        LOGS / "training_diagnostics",
        f"a2c_baseline_{RUN_TAG}_*.csv",
    )
    if diagnostics_path is not None:
        summary, warnings = diagnose_training_file(diagnostics_path)
        print_training_diagnostics_summary(diagnostics_path, summary, warnings)
        print("=" * 60)
    else:
        print("Training diagnostics: no matching CSV found.")
        print("=" * 60)
    env.close()


if __name__ == "__main__":
    main()
