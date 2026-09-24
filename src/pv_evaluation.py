"""A2C baseline與MFN-A2C的PV reward模型共用回測流程。"""

from pathlib import Path

import pandas as pd
from stable_baselines3 import A2C

# 載入自訂特徵擷取器，使SB3反序列化MFN模型時可解析類別。
from src.mfn_github_extractor import GitHubStyleTwoViewMFN  # noqa: F401
from src.evaluation_metrics import (
    add_dsr_columns,
    add_strength_alignment_columns,
    add_test_timestamps,
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
from src.experiment_config import DATA, LOGS, MODELS, RESULTS
from src.experiment_periods import LOOKBACK, PERIODS_PER_YEAR
from src.pv_experiment import make_pv_env
from src.training_diagnostics import (
    diagnose_training_file,
    latest_diagnostics,
    print_training_diagnostics_summary,
)


def evaluate_pv_model(
    *,
    title: str,
    model_name: str,
    result_name: str,
    diagnostics_pattern: str,
) -> None:
    """以deterministic policy回測指定PV reward模型並輸出完整診斷。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    test_raw = pd.read_csv(DATA / "merged_output_test.csv")
    test_period = validate_test_period(test_raw, LOOKBACK)
    env = make_pv_env("test")
    model_path = MODELS / model_name
    model = A2C.load(str(model_path), env=env, device="cpu")

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    result = add_test_timestamps(env.get_results(), test_raw, LOOKBACK)
    result = add_dsr_columns(
        result,
        eta=env.eta,
        warmup_steps=env.dsr_warmup_steps,
        formula=env.dsr_formula,
    )
    result = add_strength_alignment_columns(result)
    result.to_csv(RESULTS / result_name, index=False)

    values = result["portfolio_value"].astype(float)
    initial = values.iloc[0]
    final = values.iloc[-1]
    total_return = (final / initial - 1.0) * 100.0
    peak = values.max()
    max_drawdown = (values / values.cummax() - 1.0).min() * 100.0
    returns = values.pct_change().dropna()
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * PERIODS_PER_YEAR**0.5

    dsr_metrics = summarize_dsr(result)
    allocation_metrics = summarize_allocations(result)
    contribution_metrics = summarize_asset_contributions(result)
    strength_metrics = summarize_strength_alignment(result)

    print()
    print("=" * 68)
    print(title)
    print("=" * 68)
    print_test_period(test_period)
    print("Training reward : absolute Portfolio Value")
    print(f"Initial PV      : {initial:.2f}")
    print(f"Final PV        : {final:.2f}")
    print(f"Total Return    : {total_return:.2f}%")
    print(f"Peak PV         : {peak:.2f}")
    print(f"Max Drawdown    : {max_drawdown:.2f}%")
    print(f"Sharpe Ratio    : {sharpe:.4f}")
    print(
        "Peak Cumulative DSR  : "
        f"{dsr_metrics['Peak Cumulative DSR']:.4f}"
    )
    print(
        "Final Cumulative DSR : "
        f"{dsr_metrics['Final Cumulative DSR']:.4f}"
    )
    print("=" * 68)
    print_allocation_summary(allocation_metrics)
    print_asset_contribution_summary(contribution_metrics)
    print_strength_alignment_summary(strength_metrics)
    print("=" * 68)

    diagnostics_path = latest_diagnostics(
        LOGS / "training_diagnostics",
        diagnostics_pattern,
    )
    if diagnostics_path is not None:
        summary, warnings = diagnose_training_file(diagnostics_path)
        print_training_diagnostics_summary(diagnostics_path, summary, warnings)
        print("=" * 68)
    else:
        print("Training diagnostics: no matching CSV found.")
        print("=" * 68)

    metrics = {
        "Initial PV": initial,
        "Peak PV": peak,
        "Final PV": final,
        "Return (%)": total_return,
        "Max Drawdown (%)": max_drawdown,
        "Sharpe": sharpe,
        **dsr_metrics,
        **allocation_metrics,
        **contribution_metrics,
        **strength_metrics,
    }
    metrics_name = result_name.replace("_results.csv", "_metrics.csv")
    pd.DataFrame([metrics]).to_csv(RESULTS / metrics_name, index=False)
    print(f"Saved results: {RESULTS / result_name}")
    print(f"Saved metrics: {RESULTS / metrics_name}")
    env.close()
