"""評估4H Experiment 2的DQN baseline。"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config_4h as cfg
from evaluate_common import attach_timestamps, print_strength
from src.discrete_action_env import DiscretePortfolioWrapper
from src.evaluation_metrics import (
    add_dsr_columns, add_strength_alignment_columns,
    print_allocation_summary, print_asset_contribution_summary,
    summarize_allocations, summarize_asset_contributions,
    summarize_dsr, summarize_strength_alignment,
)


def main() -> None:
    cfg.MODEL_RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(cfg.DATA / "merged_output_test.csv")
    base_env = cfg.make_portfolio_env("test", action_mode=cfg.DQN_ACTION_MODE)
    env = DiscretePortfolioWrapper(
        base_env,
        weight_step=cfg.DQN_WEIGHT_STEP,
        max_crypto_weight=cfg.DQN_MAX_CRYPTO_WEIGHT,
    )
    model = DQN.load(str(cfg.MODELS / cfg.DQN_MODEL_NAME), env=env, device="cpu")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

    result = attach_timestamps(base_env.get_results(), raw)
    result = add_dsr_columns(result, eta=base_env.eta,
                             warmup_steps=base_env.dsr_warmup_steps,
                             formula=base_env.dsr_formula)
    result = add_strength_alignment_columns(result, horizons=(3, 6, 18, 42))
    result.to_csv(cfg.MODEL_RESULTS / cfg.DQN_RESULT_NAME, index=False)
    values = pd.to_numeric(result["portfolio_value"])
    returns = values.pct_change().dropna()
    metrics = {
        "Method": "dqn", "Initial PV": float(values.iloc[0]),
        "Final PV": float(values.iloc[-1]),
        "Total Return": float(values.iloc[-1] / values.iloc[0] - 1),
        "Peak PV": float(values.max()),
        "Max Drawdown": float((values / values.cummax() - 1).min()),
        "Sharpe Ratio": float(returns.mean() / returns.std() * np.sqrt(cfg.PERIODS_PER_YEAR)) if returns.std() > 0 else 0.0,
    }
    allocation = summarize_allocations(result)
    contribution = summarize_asset_contributions(result)
    strength = summarize_strength_alignment(result, horizons=(3, 6, 18, 42))
    metrics.update(summarize_dsr(result)); metrics.update(allocation)
    metrics.update(contribution); metrics.update(strength)
    pd.DataFrame([metrics]).to_csv(
        cfg.MODEL_RESULTS / cfg.DQN_RESULT_NAME.replace("_results.csv", "_metrics.csv"),
        index=False,
    )
    print("=" * 72); print("4H EXPERIMENT 2 - DQN BACKTEST"); print("=" * 72)
    print(f"Initial PV   : {metrics['Initial PV']:.2f}")
    print(f"Final PV     : {metrics['Final PV']:.2f}")
    print(f"Total Return : {metrics['Total Return']:.2%}")
    print(f"Peak PV      : {metrics['Peak PV']:.2f}")
    print(f"Max Drawdown : {metrics['Max Drawdown']:.2%}")
    print(f"Sharpe Ratio : {metrics['Sharpe Ratio']:.4f}")
    print_allocation_summary(allocation)
    print_asset_contribution_summary(contribution)
    print_strength(strength)
    env.close()


if __name__ == "__main__":
    main()
