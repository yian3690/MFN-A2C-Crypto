"""Backtest the discrete-action DQN baseline on the held-out test period."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN

from src.experiment_config import (
    DATA,
    DQN_ACTION_MODE,
    DQN_MAX_CRYPTO_WEIGHT,
    DQN_MODEL_NAME,
    DQN_RESULT_NAME,
    DQN_WEIGHT_STEP,
    MODELS,
    RESULTS,
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
from src.discrete_action_env import DiscretePortfolioWrapper
from src.experiment_periods import LOOKBACK, PERIODS_PER_YEAR

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    """主程式入口：依序執行此腳本定義的完整流程。"""
    test_raw = pd.read_csv(
        DATA / "merged_output_test.csv"
    )
    test_period = validate_test_period(test_raw, LOOKBACK)
    base_env = make_portfolio_env("test", action_mode=DQN_ACTION_MODE)

    env = DiscretePortfolioWrapper(
        base_env,
        weight_step=DQN_WEIGHT_STEP,
        max_crypto_weight=DQN_MAX_CRYPTO_WEIGHT,
    )

    model = DQN.load(
        str(MODELS / DQN_MODEL_NAME),
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
            terminated or truncated
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
        DQN_RESULT_NAME
    )

    result.to_csv(
        output,
        index=False,
    )

    pv = (
        result["portfolio_value"]
        .astype(float)
    )

    initial = pv.iloc[0]
    final = pv.iloc[-1]
    peak = pv.max()

    total_return = (
        final / initial - 1
    ) * 100

    running_max = pv.cummax()

    drawdown = (
        pv / running_max - 1
    )

    max_dd = (
        drawdown.min() * 100
    )

    returns = pv.pct_change().dropna()

    if returns.std() > 0:

        # 12 × 365 two-hour periods per year
        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(PERIODS_PER_YEAR)
        )

    else:
        sharpe = 0.0

    print()
    print("=" * 60)
    print("DQN BASELINE RESULTS")
    print("=" * 60)
    print_test_period(test_period)

    print(
        f"Initial PV   : {initial:.2f}"
    )

    print(
        f"Final PV     : {final:.2f}"
    )

    print(
        f"Peak PV      : {peak:.2f}"
    )

    print(
        f"Total Return : {total_return:.2f}%"
    )

    print(
        f"Max Drawdown : {max_dd:.2f}%"
    )

    print(
        f"Sharpe       : {sharpe:.4f}"
    )

    print(
        f"Peak Cumulative DSR     : {dsr_metrics['Peak Cumulative DSR']:.4f}"
    )

    print(
        f"Final Cumulative DSR    : {dsr_metrics['Final Cumulative DSR']:.4f}"
    )

    print()
    print_allocation_summary(allocation_metrics)
    print_asset_contribution_summary(contribution_metrics)
    print_strength_alignment_summary(strength_metrics)
    print()
    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()
