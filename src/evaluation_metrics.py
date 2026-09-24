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
            "and scripts/prepare_features_4h.py before evaluation."
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


def summarize_asset_contributions(result: pd.DataFrame) -> dict[str, float]:
    """計算各資產對最終PV變化的路徑相依損益貢獻。

    每一步先以交易前PV乘上 ``weight_i * return_i`` 得到資產損益，
    再沿回測期間加總。因環境不含費用，各資產損益總和應精確等於
    ``Final PV - Initial PV``；這比直接相加百分比報酬更適合解釋PV。
    """
    assets = ("BTC", "ETH", "LTC", "BNB", "USDT")
    pnl_columns = [f"pnl_contribution_{asset.lower()}" for asset in assets]
    return_columns = [
        f"return_contribution_{asset.lower()}" for asset in assets
    ]
    asset_return_columns = [
        f"asset_return_{asset.lower()}" for asset in assets
    ]
    missing = [
        column
        for column in pnl_columns + return_columns + asset_return_columns
        if column not in result
    ]
    if missing:
        raise ValueError(f"Missing asset-contribution columns: {missing}")

    initial_pv = float(result["portfolio_value"].iloc[0])
    final_pv = float(result["portfolio_value"].iloc[-1])
    total_pnl = final_pv - initial_pv
    metrics: dict[str, float] = {
        "Total Attributed PnL": total_pnl,
    }
    attributed_total = 0.0
    for asset, pnl_column, return_column, asset_return_column in zip(
        assets,
        pnl_columns,
        return_columns,
        asset_return_columns,
    ):
        pnl = float(pd.to_numeric(result[pnl_column], errors="coerce").sum())
        simple_return = float(
            pd.to_numeric(result[return_column], errors="coerce").sum()
        )
        asset_returns = pd.to_numeric(
            result[asset_return_column],
            errors="coerce",
        ).dropna()
        buy_hold_return = float((1.0 + asset_returns).prod() - 1.0)
        share = pnl / total_pnl if not np.isclose(total_pnl, 0.0) else np.nan
        metrics[f"{asset} PnL Contribution"] = pnl
        metrics[f"{asset} PnL Share"] = share
        metrics[f"{asset} Summed Return Contribution"] = simple_return
        metrics[f"{asset} Underlying Return"] = buy_hold_return
        attributed_total += pnl

    metrics["Attribution Reconciliation Error"] = attributed_total - total_pnl
    return metrics


def print_asset_contribution_summary(metrics: dict[str, float]) -> None:
    """輸出逐資產PV損益歸因及加總核對誤差。"""
    print("Asset return contribution:")
    for asset in ("BTC", "ETH", "LTC", "BNB", "USDT"):
        pnl = metrics[f"{asset} PnL Contribution"]
        share = metrics[f"{asset} PnL Share"]
        simple_return = metrics[f"{asset} Summed Return Contribution"]
        underlying_return = metrics[f"{asset} Underlying Return"]
        share_text = "n/a" if np.isnan(share) else f"{share:7.2%}"
        print(
            f"  {asset:<4}: asset return {underlying_return:+8.2%} | "
            f"PnL {pnl:+10.2f} USDT | share {share_text} | "
            f"summed step contribution "
            f"{simple_return:+8.2%}"
        )
    print(
        "  Total attributed PnL : "
        f"{metrics['Total Attributed PnL']:+.2f} USDT"
    )
    print(
        "  Reconciliation error : "
        f"{metrics['Attribution Reconciliation Error']:+.8f} USDT"
    )


def add_strength_alignment_columns(
    result: pd.DataFrame,
    horizons: tuple[int, ...] = (6, 12, 36, 84),
) -> pd.DataFrame:
    """加入權重是否跟隨已知相對強勢資產的事後診斷欄位。

    Trailing momentum一律先shift(1)，只使用動作當下已完成的報酬，
    不會把本期Open[t]到Open[t+1]的未來報酬混入動量。當期實現
    winner權重則明確標為ex-post診斷，只能解釋結果，不可作為特徵。
    """
    output = result.copy()
    assets = ("btc", "eth", "ltc", "bnb", "usdt")
    weight_columns = [f"weight_{asset}" for asset in assets]
    return_columns = [f"asset_return_{asset}" for asset in assets]
    missing = [
        column
        for column in weight_columns + return_columns
        if column not in output
    ]
    if missing:
        raise ValueError(f"Missing strength-alignment columns: {missing}")

    weights = output[weight_columns].apply(pd.to_numeric, errors="coerce")
    asset_returns = output[return_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    # 統一欄名後再做逐列運算，避免pandas依原始欄名對齊成全NaN。
    weights.columns = assets
    asset_returns.columns = assets

    for horizon in horizons:
        momentum = (
            (1.0 + asset_returns)
            .shift(1)
            .rolling(horizon, min_periods=horizon)
            .apply(np.prod, raw=True)
            - 1.0
        )
        for asset in assets:
            output[f"trailing_return_{horizon}_{asset}"] = momentum[asset]

        weight_ranks = weights.rank(axis=1, method="average")
        momentum_ranks = momentum.rank(axis=1, method="average")
        weight_centered = weight_ranks.sub(weight_ranks.mean(axis=1), axis=0)
        momentum_centered = momentum_ranks.sub(
            momentum_ranks.mean(axis=1), axis=0
        )
        numerator = (weight_centered * momentum_centered).sum(axis=1)
        denominator = np.sqrt(
            weight_centered.pow(2).sum(axis=1)
            * momentum_centered.pow(2).sum(axis=1)
        )
        output[f"strength_rank_corr_{horizon}"] = numerator / denominator.replace(
            0.0, np.nan
        )

        valid = momentum.notna().all(axis=1) & weights.notna().all(axis=1)
        winner_weight = pd.Series(np.nan, index=output.index, dtype=float)
        loser_weight = pd.Series(np.nan, index=output.index, dtype=float)
        if valid.any():
            valid_momentum = momentum.loc[valid].to_numpy()
            valid_weights = weights.loc[valid].to_numpy()
            rows = np.arange(len(valid_weights))
            winner_weight.loc[valid] = valid_weights[
                rows, np.argmax(valid_momentum, axis=1)
            ]
            loser_weight.loc[valid] = valid_weights[
                rows, np.argmin(valid_momentum, axis=1)
            ]
        output[f"weight_on_trailing_winner_{horizon}"] = winner_weight
        output[f"weight_on_trailing_loser_{horizon}"] = loser_weight

    valid_realized = asset_returns.notna().all(axis=1) & weights.notna().all(axis=1)
    realized_winner_weight = pd.Series(np.nan, index=output.index, dtype=float)
    realized_loser_weight = pd.Series(np.nan, index=output.index, dtype=float)
    if valid_realized.any():
        realized = asset_returns.loc[valid_realized].to_numpy()
        realized_weights = weights.loc[valid_realized].to_numpy()
        rows = np.arange(len(realized_weights))
        realized_winner_weight.loc[valid_realized] = realized_weights[
            rows, np.argmax(realized, axis=1)
        ]
        realized_loser_weight.loc[valid_realized] = realized_weights[
            rows, np.argmin(realized, axis=1)
        ]
    output["weight_on_realized_winner"] = realized_winner_weight
    output["weight_on_realized_loser"] = realized_loser_weight
    return output


def summarize_strength_alignment(
    result: pd.DataFrame,
    horizons: tuple[int, ...] = (6, 12, 36, 84),
) -> dict[str, float]:
    """彙整多個動量期間的配置方向與事後winner exposure。"""
    metrics: dict[str, float] = {}
    for horizon in horizons:
        for prefix, label in (
            ("strength_rank_corr", "Rank Correlation"),
            ("weight_on_trailing_winner", "Trailing Winner Weight"),
            ("weight_on_trailing_loser", "Trailing Loser Weight"),
        ):
            column = f"{prefix}_{horizon}"
            values = pd.to_numeric(result[column], errors="coerce").dropna()
            metrics[f"{horizon}-step {label}"] = (
                float(values.mean()) if len(values) else np.nan
            )
    metrics["Realized Winner Weight"] = float(
        pd.to_numeric(
            result["weight_on_realized_winner"], errors="coerce"
        ).mean()
    )
    metrics["Realized Loser Weight"] = float(
        pd.to_numeric(
            result["weight_on_realized_loser"], errors="coerce"
        ).mean()
    )
    return metrics


def print_strength_alignment_summary(metrics: dict[str, float]) -> None:
    """輸出模型是否將較高權重配置給近期強勢資產。"""
    print("Relative-strength allocation diagnostics:")
    for horizon, label in ((6, "12h"), (12, "1d"), (36, "3d"), (84, "7d")):
        correlation = metrics[f"{horizon}-step Rank Correlation"]
        winner = metrics[f"{horizon}-step Trailing Winner Weight"]
        loser = metrics[f"{horizon}-step Trailing Loser Weight"]
        print(
            f"  {label:<3}: rank corr {correlation:+.3f} | "
            f"winner weight {winner:6.2%} | loser weight {loser:6.2%}"
        )
    print(
        "  Ex-post next-interval winner weight: "
        f"{metrics['Realized Winner Weight']:.2%}"
    )
    print(
        "  Ex-post next-interval loser weight : "
        f"{metrics['Realized Loser Weight']:.2%}"
    )
