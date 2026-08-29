"""Backtest the formal MFN-A2C model and export paper-style metrics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stable_baselines3 import A2C

from src.mfn_sb3_extractor import TwoViewMFN
from src.portfolio_env_sb3 import CryptoPortfolioEnv


DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


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

    # 4-hour annualization
    periods_per_year = 6 * 365

    returns = values[1:] / values[:-1] - 1.0

    if len(returns) > 1 and returns.std() > 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(periods_per_year)
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
            )) - 20
        ),

        reward_type="dsr",

        eta=0.005,

        initial_balance=10000,

        random_start=False,
    )

    model = A2C.load(

        str(MODELS / "mfn_a2c_formal"),

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

    result = env.get_results()

    result.to_csv(
        RESULTS / "formal_backtest_results.csv",
        index=False,
    )

    metrics = calculate_metrics(
        result["portfolio_value"].values
    )

    print()
    print("=" * 60)
    print("MFN-A2C FORMAL BACKTEST")
    print("=" * 60)

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

    pd.DataFrame(
        [metrics]
    ).to_csv(
        RESULTS / "formal_metrics.csv",
        index=False,
    )

    env.close()


if __name__ == "__main__":
    main()