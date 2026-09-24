"""建立論文特徵並切成32,444筆Train與1,080筆Test。"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_scaling import FeatureStandardizer
from src.feature_schema import (
    CRYPTO_ASSETS,
    INDICATOR_DIM,
    PRICE_DIM,
    RELATIVE_STRENGTH_BARS,
    RELATIVE_STRENGTH_NAME,
)
from src.experiment_periods import (
    DATA_START,
    EXPECTED_TOTAL_VALID_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALID_START,
    TEST_END_EXCLUSIVE,
    TEST_ROWS,
    TEST_START,
)


DATA = ROOT / "data"
MERGED = DATA / "merged_output.csv"
ASSETS = list(CRYPTO_ASSETS)


def main():
    """建立5維價格、25維技術指標及論文Train/Test切分。"""
    if not MERGED.exists():
        raise FileNotFoundError(
            f"Cannot find {MERGED}. Run download_binance_paper.py first."
        )
    DATA.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(MERGED)
    required = ["Open Time"] + [
        f"{field}{i}"
        for i in range(4)
        for field in ("Open", "High", "Low", "Close")
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw["Open Time"] = pd.to_datetime(raw["Open Time"], utc=True)
    raw = (
        raw.sort_values("Open Time")
        .drop_duplicates("Open Time", keep="first")
        .reset_index(drop=True)
    )
    raw = raw[
        (raw["Open Time"] >= DATA_START)
        & (raw["Open Time"] < TEST_END_EXCLUSIVE)
    ].reset_index(drop=True)

    price = pd.DataFrame({"Open Time": raw["Open Time"]})
    tech = pd.DataFrame({"Open Time": raw["Open Time"]})
    crypto_closes = pd.DataFrame(index=raw.index)
    for index, asset in enumerate(ASSETS):
        crypto_closes[asset] = pd.to_numeric(
            raw[f"Close{index}"], errors="coerce"
        )

    # RS_14D只使用當下及過去價格；最前段以當時可取得的最長歷史補足。
    rs_reference = crypto_closes.shift(RELATIVE_STRENGTH_BARS)
    rs_reference = rs_reference.fillna(crypto_closes.iloc[0])
    momentum = crypto_closes / rs_reference - 1.0
    relative_strength = momentum.sub(momentum.mean(axis=1), axis=0)

    for index, asset in enumerate(ASSETS):
        close = pd.to_numeric(raw[f"Close{index}"], errors="coerce")
        price[f"{asset}_price_relative"] = close / close.shift(1)

        sma = ta.sma(close, length=20)
        ema = ta.ema(close, length=20)
        rsi = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if (
            any(value is None for value in (sma, ema, rsi, macd))
            or "MACD_12_26_9" not in macd.columns
        ):
            raise RuntimeError(f"Technical-indicator calculation failed for {asset}")

        tech[f"{asset}_SMA20"] = sma
        tech[f"{asset}_EMA20"] = ema
        tech[f"{asset}_MACD"] = macd["MACD_12_26_9"]
        tech[f"{asset}_RSI14"] = rsi
        tech[f"{asset}_{RELATIVE_STRENGTH_NAME}"] = relative_strength[asset]

    price["USDT_price_relative"] = 1.0
    tech["USDT_SMA20"] = 1.0
    tech["USDT_EMA20"] = 1.0
    tech["USDT_MACD"] = 0.0
    tech["USDT_RSI14"] = 50.0
    tech[f"USDT_{RELATIVE_STRENGTH_NAME}"] = 0.0

    combined = price.merge(tech, on="Open Time", how="inner", validate="one_to_one")
    combined = (
        combined.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    if len(combined) < EXPECTED_TOTAL_VALID_ROWS:
        raise RuntimeError(
            f"Only {len(combined)} valid rows remain; expected at least "
            f"{EXPECTED_TOTAL_VALID_ROWS}."
        )
    combined = combined.tail(EXPECTED_TOTAL_VALID_ROWS).reset_index(drop=True)

    price_cols = [
        column for column in combined.columns if column.endswith("_price_relative")
    ]
    tech_cols = [
        column
        for column in combined.columns
        if column != "Open Time" and column not in price_cols
    ]
    if len(price_cols) != PRICE_DIM or len(tech_cols) != INDICATOR_DIM:
        raise RuntimeError(
            f"Expected {PRICE_DIM}+{INDICATOR_DIM} features, "
            f"got {len(price_cols)}+{len(tech_cols)}"
        )

    raw_indexed = raw.set_index("Open Time")
    aligned_raw = raw_indexed.loc[combined["Open Time"]].reset_index()
    if not aligned_raw["Open Time"].equals(combined["Open Time"]):
        raise RuntimeError("FINAL ALIGNMENT FAILED: raw and feature timestamps differ.")

    train_mask = (combined["Open Time"] < TEST_START).to_numpy()
    test_mask = (
        (combined["Open Time"] >= TEST_START)
        & (combined["Open Time"] < TEST_END_EXCLUSIVE)
    ).to_numpy()
    train_rows = int(train_mask.sum())
    test_rows = int(test_mask.sum())
    if len(combined) != EXPECTED_TOTAL_VALID_ROWS:
        raise RuntimeError("有效資料總數不是33,524筆。")
    if train_rows != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_ROWS} Train rows, got {train_rows}."
        )
    if test_rows != TEST_ROWS:
        raise RuntimeError(f"Expected {TEST_ROWS} Test rows, got {test_rows}.")
    if train_rows + test_rows != len(combined):
        raise RuntimeError("Train/Test masks do not cover all valid rows.")
    if combined["Open Time"].iloc[0] != pd.Timestamp(EXPECTED_VALID_START):
        raise RuntimeError(
            f"Expected first valid timestamp {EXPECTED_VALID_START}, got "
            f"{combined['Open Time'].iloc[0]}."
        )

    # 僅用32,444筆Train擬合標準化統計，Test完全不參與。
    price_scaler = FeatureStandardizer.fit(combined.loc[train_mask, price_cols])
    tech_scaler = FeatureStandardizer.fit(combined.loc[train_mask, tech_cols])
    scaled = combined.copy()
    scaled.loc[:, price_cols] = price_scaler.transform(combined[price_cols])
    scaled.loc[:, tech_cols] = tech_scaler.transform(combined[tech_cols])

    scaler_table = pd.concat(
        [
            price_scaler.to_frame("price"),
            tech_scaler.to_frame("technical_indicator"),
        ],
        ignore_index=True,
    )
    scaler_table.to_csv(DATA / "feature_scaler_train.csv", index=False)
    scaler_table.to_csv(DATA / "feature_scaler.csv", index=False)

    for split_name, mask in (("train", train_mask), ("test", test_mask)):
        features = scaled.loc[mask].reset_index(drop=True)
        raw_split = aligned_raw.loc[mask].reset_index(drop=True)
        features[["Open Time"]].to_csv(
            DATA / f"timestamps_{split_name}.csv", index=False
        )
        features[price_cols].to_csv(
            DATA / f"pct_change_output_{split_name}.csv", index=False
        )
        features[tech_cols].to_csv(
            DATA / f"ta_test_{split_name}.csv", index=False
        )
        raw_split.to_csv(DATA / f"merged_output_{split_name}.csv", index=False)
        if not features["Open Time"].equals(raw_split["Open Time"]):
            raise RuntimeError(f"{split_name}: timestamp mismatch")

    # 移除舊三段切分產物，確保data目錄只保留Train/Test版本。
    for legacy_split in ("validation", "development"):
        for prefix in (
            "timestamps",
            "pct_change_output",
            "ta_test",
            "merged_output",
        ):
            (DATA / f"{prefix}_{legacy_split}.csv").unlink(missing_ok=True)
    (DATA / "feature_scaler_development.csv").unlink(missing_ok=True)

    print("Feature preparation complete.")
    print(f"Total valid rows : {len(combined):,}")
    print(f"Train rows       : {train_rows:,}")
    print(f"Test rows        : {test_rows:,}")
    print(f"Price features   : {len(price_cols)}")
    print(f"TA features      : {len(tech_cols)}")
    print(
        "Train period     : "
        f"{combined.loc[train_mask, 'Open Time'].iloc[0]} -> "
        f"{combined.loc[train_mask, 'Open Time'].iloc[-1]}"
    )
    print(
        "Test period      : "
        f"{combined.loc[test_mask, 'Open Time'].iloc[0]} -> "
        f"{combined.loc[test_mask, 'Open Time'].iloc[-1]}"
    )
    print(f"Feature levels   : SMA / EMA / MACD(DIF) / RSI / {RELATIVE_STRENGTH_NAME}")
    print("Scaling          : Train-only z-score")
    print("Alignment check  : PASS")


if __name__ == "__main__":
    main()
