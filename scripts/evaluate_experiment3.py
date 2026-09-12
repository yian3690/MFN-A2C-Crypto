"""Evaluate all reward-function configurations used in Experiment 3."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import A2C

from src.portfolio_env_sb3 import CryptoPortfolioEnv
from src.mfn_sb3_extractor import TwoViewMFN


# ============================================================
# Paths
# ============================================================

DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Environment
# ============================================================

def make_test_env(reward_type):

    """Create a deterministic held-out environment for one reward setting."""
    raw_test = pd.read_csv(
        DATA / "merged_output_test.csv"
    )

    max_steps = (
        len(raw_test) - 20 - 1
    )

    env = CryptoPortfolioEnv(

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

        n_previous_timesteps=20,

        max_episode_steps=max_steps,

        reward_type=reward_type,

        initial_balance=10000.0,

        eta=0.005,

        random_start=False,
    )

    return env


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(result):

    """Calculate the backtest metrics saved in the Experiment 3 summary."""
    pv = (
        result["portfolio_value"]
        .astype(float)
        .reset_index(drop=True)
    )

    initial = pv.iloc[0]

    final = pv.iloc[-1]

    peak = pv.max()

    total_return = (
        final /
        initial -
        1
    ) * 100

    running_max = pv.cummax()

    drawdown = (
        pv /
        running_max -
        1
    )

    max_drawdown = (
        drawdown.min()
        * 100
    )

    returns = (
        pv.pct_change()
        .dropna()
    )

    if (
        len(returns) > 1
        and returns.std() > 0
    ):

        sharpe = (
            returns.mean()
            /
            returns.std()
            *
            np.sqrt(
                6 * 365
            )
        )

    else:

        sharpe = 0.0

    return {
        "Initial PV":
            initial,

        "Peak PV":
            peak,

        "Final PV":
            final,

        "Return (%)":
            total_return,

        "Max Drawdown (%)":
            max_drawdown,

        "Sharpe":
            sharpe,
    }


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_model(
    model_name,
    reward_type,
    output_name,
):

    """Load one trained model, backtest it, save its curve, and return its metrics."""
    print()
    print("=" * 70)
    print(
        f"Evaluating: {model_name}"
    )
    print("=" * 70)

    env = make_test_env(
        reward_type
    )

    model_path = (
        MODELS /
        model_name
    )

    model = A2C.load(
        str(model_path),
        device="cpu",
    )

    obs, info = env.reset()

    terminated = False
    truncated = False

    while not (
        terminated
        or truncated
    ):

        action, _ = (
            model.predict(
                obs,
                deterministic=True,
            )
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

    result = env.get_results()

    output_path = (
        RESULTS /
        output_name
    )

    result.to_csv(
        output_path,
        index=False,
    )

    metrics = (
        calculate_metrics(
            result
        )
    )

    print()

    for key, value in (
        metrics.items()
    ):

        print(
            f"{key:<20}: "
            f"{value:.4f}"
        )

    print()
    print(
        f"Saved: {output_path}"
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    """主程式入口：依序執行此腳本定義的完整流程。"""
    all_metrics = {}

    # --------------------------------------------------------
    # MFN-A2C + DSR
    # --------------------------------------------------------

    all_metrics[
        "MFN-A2C + DSR"
    ] = evaluate_model(

        model_name=
            "exp3_mfn_dsr",

        reward_type=
            "dsr",

        output_name=
            "exp3_mfn_dsr_results.csv",
    )

    # --------------------------------------------------------
    # MFN-A2C + PV
    # --------------------------------------------------------

    all_metrics[
        "MFN-A2C + PV"
    ] = evaluate_model(

        model_name=
            "exp3_mfn_pv",

        reward_type=
            "pv",

        output_name=
            "exp3_mfn_pv_results.csv",
    )

    # --------------------------------------------------------
    # A2C + DSR
    # --------------------------------------------------------

    all_metrics[
        "A2C + DSR"
    ] = evaluate_model(

        model_name=
            "exp3_a2c_dsr",

        reward_type=
            "dsr",

        output_name=
            "exp3_a2c_dsr_results.csv",
    )

    # --------------------------------------------------------
    # A2C + PV
    # --------------------------------------------------------

    all_metrics[
        "A2C + PV"
    ] = evaluate_model(

        model_name=
            "exp3_a2c_pv",

        reward_type=
            "pv",

        output_name=
            "exp3_a2c_pv_results.csv",
    )

    # --------------------------------------------------------
    # Save metrics table
    # --------------------------------------------------------

    metrics_df = (
        pd.DataFrame(
            all_metrics
        ).T
    )

    metrics_output = (
        RESULTS /
        "experiment3_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_output
    )

    print()
    print("=" * 80)
    print(
        "EXPERIMENT 3 SUMMARY"
    )
    print("=" * 80)

    print(
        metrics_df.round(4)
    )

    print()
    print(
        f"Saved: {metrics_output}"
    )


if __name__ == "__main__":
    main()