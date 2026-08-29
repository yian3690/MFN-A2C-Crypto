"""
Backtest a trained MFN-A2C model on the final 1,080 rows.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from src.mfn_sb3_extractor import TwoViewMFN
from src.portfolio_env_sb3 import CryptoPortfolioEnv


DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    env = CryptoPortfolioEnv(
        pct_csv=str(DATA / "pct_change_output_test.csv"),
        ta_csv=str(DATA / "ta_test_test.csv"),
        raw_csv=str(DATA / "merged_output_test.csv"),
        n_previous_timesteps=20,
        max_episode_steps=len(pd.read_csv(DATA / "pct_change_output_test.csv")) - 20 - 1,
        reward_type="dsr",
        initial_balance=10000,
        eta=0.005,
        random_start=False,
    )
    env = Monitor(env)

    model = A2C.load(str(MODELS / "mfn_a2c_test"), env=env)

    obs, info = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    result = env.unwrapped.get_results()
    result.to_csv(RESULTS / "backtest_results.csv", index=False)

    print("\nFinal portfolio value:", result["portfolio_value"].iloc[-1])
    print("Peak portfolio value :", result["portfolio_value"].max())

    plt.figure(figsize=(10, 5))
    plt.plot(result["portfolio_value"])
    plt.title("MFN-A2C Backtest Portfolio Value")
    plt.xlabel("4-hour timestep")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.savefig(FIGURES / "backtest_portfolio_value.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
