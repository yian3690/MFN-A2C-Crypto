"""Backtest the discrete-action DQN baseline on the held-out test period."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN

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
from src.discrete_action_env import DiscretePortfolioWrapper
from src.experiment_periods import PERIODS_PER_YEAR


DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MAX_CRYPTO_WEIGHT = 0.60

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    """主程式入口：依序執行此腳本定義的完整流程。"""
    test_raw = pd.read_csv(
        DATA / "merged_output_test.csv"
    )
    test_period = validate_test_period(test_raw, 20)

    max_steps = (
        len(test_raw) - 20 - 1
    )

    base_env = CryptoPortfolioEnv(
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

        max_episode_steps=max_steps,

        reward_type="dsr",

        initial_balance=10000.0,

        eta=0.005,

        random_start=False,
)

    env = DiscretePortfolioWrapper(
        base_env,
        weight_step=0.2,
        max_crypto_weight=MAX_CRYPTO_WEIGHT,
    )

    model = DQN.load(
        str(MODELS / "dqn_baseline"),
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
        20,
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
        "dqn_baseline_results.csv"
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
    print()
    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()
