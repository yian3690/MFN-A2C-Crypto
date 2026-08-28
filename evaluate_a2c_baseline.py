from pathlib import Path

import pandas as pd

from stable_baselines3 import A2C

from portfolio_env_sb3 import CryptoPortfolioEnv


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

LOOKBACK = 20


def main():

    test_file = DATA / "pct_change_output_test.csv"

    test_rows = len(
        pd.read_csv(test_file)
    )

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

        n_previous_timesteps=LOOKBACK,

        max_episode_steps=test_rows - LOOKBACK,

        reward_type="dsr",

        eta=0.005,

        initial_balance=10000,

        random_start=False,
    )

    model = A2C.load(

        str(
            ROOT / "a2c_baseline"
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

        done = terminated or truncated

    result = env.get_results()

    output = (
        ROOT /
        "a2c_baseline_results.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    values = result[
        "portfolio_value"
    ].astype(float)

    initial = values.iloc[0]
    final = values.iloc[-1]

    total_return = (
        final / initial - 1
    ) * 100

    peak = values.max()

    running_max = (
        values.cummax()
    )

    drawdown = (
        values /
        running_max -
        1
    )

    max_drawdown = (
        drawdown.min() * 100
    )

    returns = (
        values.pct_change()
        .dropna()
    )

    if returns.std() > 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * (6 * 365) ** 0.5
        )

    else:

        sharpe = 0.0

    print()
    print("=" * 60)
    print("A2C BASELINE BACKTEST")
    print("=" * 60)

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
        f"Sharpe Ratio    : {sharpe:.4f}"
    )

    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()