"""Backtest the price-only A2C configuration from Experiment 1."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from stable_baselines3 import A2C

from src.experiment_config import (
    A2C_ACTION_MODE,
    A2C_WITHOUT_TI_MODEL_NAME,
    A2C_WITHOUT_TI_RESULT_NAME,
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
from src.price_only_env import PriceOnlyWrapper
from src.experiment_periods import LOOKBACK, PERIODS_PER_YEAR
from src.training_diagnostics import (
    diagnose_training_file,
    latest_diagnostics,
    print_training_diagnostics_summary,
)
from src.feature_schema import PRICE_DIM


def main():
    """主程式入口：依序執行此腳本定義的完整流程。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    test_raw = pd.read_csv(DATA / "merged_output_test.csv")
    test_period = validate_test_period(test_raw, LOOKBACK)

    base_env = make_portfolio_env("test", action_mode=A2C_ACTION_MODE)

    env = PriceOnlyWrapper(
        base_env,
        price_dim=PRICE_DIM,
    )

    model = A2C.load(

        str(
            MODELS /
            A2C_WITHOUT_TI_MODEL_NAME
        ),

        env=env,

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

        done = (
            terminated or
            truncated
        )

    result = add_test_timestamps(
        base_env.get_results(),
        test_raw,
        LOOKBACK,
    )
    result = add_dsr_columns(
        result,
        eta=base_env.eta,
        warmup_steps=base_env.dsr_warmup_steps,
        formula=base_env.dsr_formula,
    )
    result = add_strength_alignment_columns(result)
    dsr_metrics = summarize_dsr(result)
    allocation_metrics = summarize_allocations(result)
    contribution_metrics = summarize_asset_contributions(result)
    strength_metrics = summarize_strength_alignment(result)

    output = (
        RESULTS /
        A2C_WITHOUT_TI_RESULT_NAME
    )

    result.to_csv(
        output,
        index=False,
    )

    values = (
        result[
            "portfolio_value"
        ]
        .astype(float)
    )

    initial = values.iloc[0]

    final = values.iloc[-1]

    peak = values.max()

    total_return = (
        final /
        initial -
        1
    ) * 100

    running_max = (
        values.cummax()
    )

    drawdown = (
        values /
        running_max -
        1
    )

    max_drawdown = (
        drawdown.min() *
        100
    )

    returns = (
        values
        .pct_change()
        .dropna()
    )

    if returns.std() > 0:

        sharpe = (
            returns.mean()
            /
            returns.std()
            *
            PERIODS_PER_YEAR ** 0.5
        )

    else:

        sharpe = 0.0

    print()
    print("=" * 60)
    print("A2C WITHOUT TI BACKTEST")
    print("=" * 60)
    print_test_period(test_period)

    print(
        f"Initial PV      : {initial:.2f}"
    )

    print(
        f"Final PV        : {final:.2f}"
    )

    print(
        f"Total Return    : {total_return:.2f}%"
    )

    print(
        f"Peak PV         : {peak:.2f}"
    )

    print(
        f"Max Drawdown    : {max_drawdown:.2f}%"
    )

    print(
        f"Sharpe Ratio          : {sharpe:.4f}"
    )

    print(
        f"Peak Cumulative DSR   : {dsr_metrics['Peak Cumulative DSR']:.4f}"
    )

    print(
        f"Final Cumulative DSR  : {dsr_metrics['Final Cumulative DSR']:.4f}"
    )

    print("=" * 60)
    print_allocation_summary(allocation_metrics)
    print_asset_contribution_summary(contribution_metrics)
    print_strength_alignment_summary(strength_metrics)
    print("=" * 60)
    diagnostics_path = latest_diagnostics(
        LOGS / "training_diagnostics",
        f"a2c_without_ti_{RUN_TAG}_*.csv",
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
