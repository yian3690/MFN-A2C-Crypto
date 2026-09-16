"""Backtest the price-only A2C configuration from Experiment 1."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from stable_baselines3 import A2C

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
from src.price_only_env import PriceOnlyWrapper
from src.experiment_periods import PERIODS_PER_YEAR
from src.feature_schema import PRICE_DIM


DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

LOOKBACK = 20


def main():
    """主程式入口：依序執行此腳本定義的完整流程。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    test_raw = pd.read_csv(DATA / "merged_output_test.csv")
    test_period = validate_test_period(test_raw, LOOKBACK)

    test_rows = len(
        pd.read_csv(
            DATA /
            "pct_change_output_test.csv"
        )
    )

    base_env = CryptoPortfolioEnv(

        pct_csv=str(
            DATA /
            "pct_change_output_test.csv"
        ),

        ta_csv=str(
            DATA /
            "ta_test_test.csv"
        ),

        raw_csv=str(
            DATA /
            "merged_output_test.csv"
        ),

        n_previous_timesteps=LOOKBACK,

        max_episode_steps=(
            test_rows - LOOKBACK - 1
        ),

        reward_type="dsr",

        eta=0.005,

        initial_balance=10000,

        random_start=False,
    )

    env = PriceOnlyWrapper(
        base_env,
        price_dim=PRICE_DIM,
    )

    model = A2C.load(

        str(
            MODELS /
            "a2c_without_ti"
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
    )
    dsr_metrics = summarize_dsr(result)
    allocation_metrics = summarize_allocations(result)

    output = (
        RESULTS /
        "a2c_without_ti_results.csv"
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
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
