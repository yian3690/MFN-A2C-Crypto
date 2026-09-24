"""建立 4H 的 5 維價格模態、25 維技術指標模態與 Train/Test 時間切分。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_4h import (
    CRYPTO_ASSETS, DATA, DATA_END_INCLUSIVE, DATA_START, INDICATOR_DIM,
    PRICE_DIM, RELATIVE_STRENGTH_BARS, RELATIVE_STRENGTH_NAME, TEST_ROWS,
)
from src.feature_scaling import FeatureStandardizer


STALE_SPLITS = ("validation", "development")


def remove_stale_split_files() -> None:
    """刪除舊版 Validation/Development 切分，避免誤用舊資料。"""
    patterns = (
        "timestamps_{split}.csv",
        "pct_change_output_{split}.csv",
        "ta_test_{split}.csv",
        "merged_output_{split}.csv",
    )
    for split in STALE_SPLITS:
        for pattern in patterns:
            path = DATA / pattern.format(split=split)
            if path.exists():
                path.unlink()


def main() -> None:
    source = DATA / "merged_output.csv"
    if not source.exists():
        raise FileNotFoundError(f"找不到 {source}，請先執行 download_data_4h.py。")

    raw = pd.read_csv(source)
    raw["Open Time"] = pd.to_datetime(raw["Open Time"], utc=True)
    raw = raw.sort_values("Open Time").drop_duplicates("Open Time")
    raw = raw[
        (raw["Open Time"] >= DATA_START)
        & (raw["Open Time"] <= DATA_END_INCLUSIVE)
    ].reset_index(drop=True)

    price = pd.DataFrame({"Open Time": raw["Open Time"]})
    tech = pd.DataFrame({"Open Time": raw["Open Time"]})
    closes = pd.DataFrame(index=raw.index)

    for index, asset in enumerate(CRYPTO_ASSETS):
        closes[asset] = pd.to_numeric(
            raw[f"Close{index}"],
            errors="coerce",
        )

    reference = closes.shift(RELATIVE_STRENGTH_BARS).fillna(closes.iloc[0])
    momentum = closes / reference - 1.0
    relative_strength = momentum.sub(momentum.mean(axis=1), axis=0)

    for index, asset in enumerate(CRYPTO_ASSETS):
        close = closes[asset]
        price[f"{asset}_price_relative"] = close / close.shift(1)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        tech[f"{asset}_SMA20"] = ta.sma(close, length=20)
        tech[f"{asset}_EMA20"] = ta.ema(close, length=20)
        tech[f"{asset}_MACD"] = macd["MACD_12_26_9"]
        tech[f"{asset}_RSI14"] = ta.rsi(close, length=14)
        tech[f"{asset}_{RELATIVE_STRENGTH_NAME}"] = relative_strength[asset]

    price["USDT_price_relative"] = 1.0
    tech["USDT_SMA20"] = 1.0
    tech["USDT_EMA20"] = 1.0
    tech["USDT_MACD"] = 0.0
    tech["USDT_RSI14"] = 50.0
    tech[f"USDT_{RELATIVE_STRENGTH_NAME}"] = 0.0

    combined = price.merge(tech, on="Open Time", validate="one_to_one")
    combined = (
        combined
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    price_cols = [
        column
        for column in combined
        if column.endswith("_price_relative")
    ]
    tech_cols = [
        column
        for column in combined
        if column != "Open Time" and column not in price_cols
    ]

    if len(price_cols) != PRICE_DIM or len(tech_cols) != INDICATOR_DIM:
        raise RuntimeError(
            f"特徵維度錯誤：{len(price_cols)} + {len(tech_cols)}"
        )
    if len(combined) <= TEST_ROWS:
        raise RuntimeError("有效資料不足以建立 Train/Test。")

    raw_indexed = raw.set_index("Open Time")
    aligned_raw = raw_indexed.loc[combined["Open Time"]].reset_index()

    total = len(combined)
    test_start = total - TEST_ROWS

    # 不再保留 Validation；原先 Test 前的 45 天也納入 Train。
    positions = {
        "train": np.arange(0, test_start),
        "test": np.arange(test_start, total),
    }

    test_times = combined.loc[
        positions["test"],
        "Open Time",
    ].reset_index(drop=True)
    differences = test_times.diff().dropna()
    if not (differences == pd.Timedelta(hours=4)).all():
        raise RuntimeError("最近 1080 筆 Test 存在非 4H 的時間缺口。")

    # scaler 僅使用 Train；Test 完全不參與 fit。
    train_rows = positions["train"]
    price_scaler = FeatureStandardizer.fit(
        combined.loc[train_rows, price_cols]
    )
    tech_scaler = FeatureStandardizer.fit(
        combined.loc[train_rows, tech_cols]
    )

    scaled = combined.copy()
    scaled.loc[:, price_cols] = price_scaler.transform(combined[price_cols])
    scaled.loc[:, tech_cols] = tech_scaler.transform(combined[tech_cols])

    scaler_table = pd.concat([
        price_scaler.to_frame("price"),
        tech_scaler.to_frame("technical_indicator"),
    ], ignore_index=True)
    scaler_table.to_csv(DATA / "feature_scaler_train.csv", index=False)

    # 清掉舊 Validation/Development CSV，避免之後誤以為仍有使用。
    remove_stale_split_files()

    for split, indexer in positions.items():
        features = scaled.iloc[indexer].reset_index(drop=True)
        raw_split = aligned_raw.iloc[indexer].reset_index(drop=True)

        features[["Open Time"]].to_csv(
            DATA / f"timestamps_{split}.csv",
            index=False,
        )
        features[price_cols].to_csv(
            DATA / f"pct_change_output_{split}.csv",
            index=False,
        )
        features[tech_cols].to_csv(
            DATA / f"ta_test_{split}.csv",
            index=False,
        )
        raw_split.to_csv(
            DATA / f"merged_output_{split}.csv",
            index=False,
        )

    print("4H feature preparation complete (Train/Test only)")
    for split, indexer in positions.items():
        times = combined.iloc[indexer]["Open Time"]
        print(
            f"{split:<11}: {len(indexer):>6,} | "
            f"{times.iloc[0]} -> {times.iloc[-1]}"
        )
    print("Validation    : disabled; former validation rows merged into Train")
    print(f"RS_14D       : {RELATIVE_STRENGTH_BARS} bars")
    print(f"Feature dims : {PRICE_DIM} + {INDICATOR_DIM}")
    print("Scaler       : Train-only z-score")


if __name__ == "__main__":
    main()
