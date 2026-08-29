"""Backtest the discrete-action DQN baseline on the held-out test period."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN

from src.portfolio_env_sb3 import CryptoPortfolioEnv
from src.discrete_action_env import DiscretePortfolioWrapper


DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    """主程式入口：依序執行此腳本定義的完整流程。"""
    test_raw = pd.read_csv(
        DATA / "merged_output_test.csv"
    )

    max_steps = (
        len(test_raw) - 20
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
        base_env
    )

    model = DQN.load(
        str(MODELS / "dqn_baseline"),
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

    result = base_env.get_results()

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

        # 6 × 365 four-hour periods per year
        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(6 * 365)
        )

    else:
        sharpe = 0.0

    print()
    print("=" * 60)
    print("DQN BASELINE RESULTS")
    print("=" * 60)

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

    print()
    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()