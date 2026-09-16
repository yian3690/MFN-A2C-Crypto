"""Shared metrics used by the model evaluation scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.dsr import (
    DEFAULT_ETA,
    DEFAULT_FORMULA,
    DEFAULT_WARMUP_STEPS,
    calculate_dsr_series,
)
from src.experiment_periods import (
    BAR_INTERVAL,
    LOOKBACK,
    TEST_END_EXCLUSIVE,
    TEST_ROWS,
    TEST_START,
)


def validate_test_period(
    raw_data: pd.DataFrame,
    lookback: int = LOOKBACK,
) -> dict[str, pd.Timestamp]:
    """Fail fast unless evaluation uses the fixed out-of-sample period."""
    if "Open Time" not in raw_data:
        raise ValueError("Test raw data is missing the Open Time column.")
    timestamps = pd.to_datetime(raw_data["Open Time"], utc=True)
    expected_end = pd.Timestamp(TEST_END_EXCLUSIVE - BAR_INTERVAL)
    if (
        len(timestamps) != TEST_ROWS
        or timestamps.iloc[0] != pd.Timestamp(TEST_START)
        or timestamps.iloc[-1] != expected_end
    ):
        raise ValueError(
            "Unexpected Test period. Run scripts/download_binance_paper.py "
            "and scripts/prepare_paper_features.py before evaluation."
        )
    if lookback >= len(timestamps):
        raise ValueError("Lookback is not shorter than the Test period.")
    return {
        "data_start": timestamps.iloc[0],
        "trading_start": timestamps.iloc[lookback],
        "data_end": timestamps.iloc[-1],
    }


def print_test_period(period: dict[str, pd.Timestamp]) -> None:
    """Print the fixed data and first tradable timestamps."""
    print(f"Test data period : {period['data_start']} -> {period['data_end']}")
    print(f"First trade      : {period['trading_start']}")


def add_test_timestamps(
    result: pd.DataFrame,
    raw_data: pd.DataFrame,
    lookback: int = LOOKBACK,
) -> pd.DataFrame:
    """Attach the exact Test timestamps represented by a result curve."""
    validate_test_period(raw_data, lookback)
    timestamps = pd.to_datetime(raw_data["Open Time"], utc=True).iloc[lookback:]
    if len(result) != len(timestamps):
        raise ValueError(
            f"Result has {len(result)} rows but Test timeline has "
            f"{len(timestamps)} rows."
        )
    output = result.copy()
    output.insert(0, "timestamp", timestamps.reset_index(drop=True))
    return output


def validate_saved_result_period(result: pd.DataFrame) -> None:
    """Reject stale result CSVs produced for an earlier Test period."""
    if "timestamp" not in result:
        raise ValueError(
            "Result CSV has no timestamp column. Retrain/re-evaluate the "
            "model for the configured Test period."
        )
    timestamps = pd.to_datetime(result["timestamp"], utc=True)
    expected_start = pd.Timestamp(TEST_START + LOOKBACK * BAR_INTERVAL)
    expected_end = pd.Timestamp(TEST_END_EXCLUSIVE - BAR_INTERVAL)
    expected_rows = TEST_ROWS - LOOKBACK
    if (
        len(timestamps) != expected_rows
        or timestamps.iloc[0] != expected_start
        or timestamps.iloc[-1] != expected_end
    ):
        raise ValueError(
            "Result CSV belongs to a different Test period. Run the "
            "corresponding evaluator again."
        )


def add_dsr_columns(
    result: pd.DataFrame,
    eta: float = DEFAULT_ETA,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    formula: str = DEFAULT_FORMULA,
) -> pd.DataFrame:
    """Add comparable step and cumulative DSR columns to a backtest."""
    output = result.copy()
    valid_returns = pd.to_numeric(
        output["return"],
        errors="coerce",
    ).dropna()

    dsr_values = calculate_dsr_series(
        valid_returns.to_numpy(),
        eta=eta,
        warmup_steps=warmup_steps,
        formula=formula,
    )

    output["dsr"] = np.nan
    output.loc[valid_returns.index, "dsr"] = dsr_values
    output["cumulative_dsr"] = output["dsr"].fillna(0.0).cumsum()
    return output


def summarize_dsr(result: pd.DataFrame) -> dict[str, float]:
    """Return paper-style peak and final cumulative DSR values."""
    cumulative = pd.to_numeric(
        result["cumulative_dsr"],
        errors="coerce",
    ).dropna()
    return {
        "Peak Cumulative DSR": float(cumulative.max()),
        "Final Cumulative DSR": float(cumulative.iloc[-1]),
    }


def summarize_allocations(result: pd.DataFrame) -> dict[str, float]:
    """Summarize model allocations and turnover, excluding the initial state."""
    weight_columns = [
        "weight_btc",
        "weight_eth",
        "weight_ltc",
        "weight_bnb",
        "weight_usdt",
    ]
    missing = [column for column in weight_columns if column not in result]
    if missing:
        raise ValueError(f"Missing allocation columns: {missing}")

    allocations = result.loc[result["return"].notna(), weight_columns]
    metrics: dict[str, float] = {}
    for column in weight_columns:
        asset = column.removeprefix("weight_").upper()
        metrics[f"Average {asset} Weight"] = float(allocations[column].mean())
        metrics[f"Maximum {asset} Weight"] = float(allocations[column].max())

    btc_eth = allocations["weight_btc"] + allocations["weight_eth"]
    metrics["Average BTC+ETH Weight"] = float(btc_eth.mean())
    metrics["BTC+ETH >= 80% Steps"] = float((btc_eth >= 0.8 - 1e-8).mean())

    turnover = pd.to_numeric(result["turnover"], errors="coerce").dropna()
    metrics["Average Turnover"] = float(turnover.mean())
    metrics["Cumulative Turnover"] = float(turnover.sum())
    return metrics


def print_allocation_summary(metrics: dict[str, float]) -> None:
    """Print a compact, consistently formatted allocation report."""
    print("Allocation averages:")
    for asset in ("BTC", "ETH", "LTC", "BNB", "USDT"):
        average = metrics[f"Average {asset} Weight"]
        maximum = metrics[f"Maximum {asset} Weight"]
        print(f"  {asset:<4}: avg {average:7.2%} | max {maximum:7.2%}")
    print(f"  BTC+ETH average       : {metrics['Average BTC+ETH Weight']:.2%}")
    print(f"  BTC+ETH >= 80% steps  : {metrics['BTC+ETH >= 80% Steps']:.2%}")
    print(f"  Average turnover      : {metrics['Average Turnover']:.2%}")
    print(f"  Cumulative turnover   : {metrics['Cumulative Turnover']:.2f}x")
