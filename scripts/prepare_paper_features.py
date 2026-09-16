"""Construct aligned price-change and technical-indicator features from raw K-lines."""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_scaling import FeatureStandardizer
from src.feature_schema import CRYPTO_ASSETS, INDICATOR_DIM, PRICE_DIM
from src.experiment_periods import (
    DATA_START,
    EXPECTED_TOTAL_VALID_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALID_START,
    TEST_END_EXCLUSIVE,
    TEST_ROWS,
    TEST_START,
)

DATA = ROOT / 'data'
MERGED = DATA / 'merged_output.csv'
ASSETS = list(CRYPTO_ASSETS)


def main():
    """Build the paper's 5-price and 20-indicator modalities and split them."""
    if not MERGED.exists():
        raise FileNotFoundError(f'Cannot find {MERGED}. Run download_binance_paper.py first.')
    DATA.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(MERGED)
    required = ['Open Time'] + [f'{x}{i}' for i in range(4) for x in ('Open','High','Low','Close')]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    raw['Open Time'] = pd.to_datetime(raw['Open Time'], utc=True)
    raw = raw.sort_values('Open Time').drop_duplicates('Open Time', keep='first').reset_index(drop=True)
    raw = raw[
        (raw['Open Time'] >= DATA_START)
        & (raw['Open Time'] < TEST_END_EXCLUSIVE)
    ].reset_index(drop=True)

    price = pd.DataFrame({'Open Time': raw['Open Time']})
    tech = pd.DataFrame({'Open Time': raw['Open Time']})

    for i, asset in enumerate(ASSETS):
        c = pd.to_numeric(raw[f'Close{i}'], errors='coerce')

        # Thesis equation 4.2: one close-to-close price relative per asset.
        # At observation timestamp t this uses only Close[t] and Close[t-1].
        price[f'{asset}_price_relative'] = c / c.shift(1)

        sma = ta.sma(c, length=20)
        ema = ta.ema(c, length=20)
        rsi = ta.rsi(c, length=14)
        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if any(v is None for v in (sma, ema, rsi, macd)) or 'MACD_12_26_9' not in macd.columns:
            raise RuntimeError(f'Technical-indicator calculation failed for {asset}')
        # Use indicator levels stated in the thesis. Train-only z-score
        # scaling below handles their different numerical units.
        tech[f'{asset}_SMA20'] = sma
        tech[f'{asset}_EMA20'] = ema
        # The archived implementation selects MACD_12_26_9 (the DIF/MACD
        # line), which also agrees with the reported 26-row warm-up.
        tech[f'{asset}_MACD'] = macd['MACD_12_26_9']
        tech[f'{asset}_RSI14'] = rsi

    # USDT is the fifth risk-free asset. Its neutral constants become zero
    # after Train-only z-score scaling and therefore add no false signal.
    price['USDT_price_relative'] = 1.0
    tech['USDT_SMA20'] = 1.0
    tech['USDT_EMA20'] = 1.0
    tech['USDT_MACD'] = 0.0
    tech['USDT_RSI14'] = 50.0

    # THE FIX: join all modalities and raw rows by the SAME timestamp before dropna.
    combined = price.merge(tech, on='Open Time', how='inner', validate='one_to_one')
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # The thesis reports exactly 33,524 valid observations. Public Binance
    # archives may contain a few more early candles than the original data
    # snapshot, so retain the most recent aligned observations to reproduce
    # the reported row-count split while keeping the final Test fixed.
    if len(combined) < EXPECTED_TOTAL_VALID_ROWS:
        raise RuntimeError(
            f'Only {len(combined)} valid rows remain; expected at least '
            f'{EXPECTED_TOTAL_VALID_ROWS}.'
        )
    combined = combined.tail(EXPECTED_TOTAL_VALID_ROWS).reset_index(drop=True)

    price_cols = [c for c in combined.columns if c.endswith('_price_relative')]
    tech_cols = [c for c in combined.columns if c != 'Open Time' and c not in price_cols]
    if len(price_cols) != PRICE_DIM or len(tech_cols) != INDICATOR_DIM:
        raise RuntimeError(
            f'Expected {PRICE_DIM}+{INDICATOR_DIM} features, '
            f'got {len(price_cols)}+{len(tech_cols)}'
        )
    # THE FIX: recover raw OHLC using exact timestamps, never tail slicing.
    raw_indexed = raw.set_index('Open Time')
    aligned_raw = raw_indexed.loc[combined['Open Time']].reset_index()
    if not aligned_raw['Open Time'].equals(combined['Open Time']):
        raise RuntimeError('FINAL ALIGNMENT FAILED: raw and feature timestamps differ.')

    train_mask = combined['Open Time'] < TEST_START
    test_mask = (
        (combined['Open Time'] >= TEST_START)
        & (combined['Open Time'] < TEST_END_EXCLUSIVE)
    )

    train_rows = int(train_mask.sum())
    test_rows = int(test_mask.sum())
    if len(combined) != EXPECTED_TOTAL_VALID_ROWS:
        raise RuntimeError(
            f'Expected {EXPECTED_TOTAL_VALID_ROWS} total valid rows, got '
            f'{len(combined)}.'
        )
    if combined['Open Time'].iloc[0] != pd.Timestamp(EXPECTED_VALID_START):
        raise RuntimeError(
            f'Expected first valid timestamp {EXPECTED_VALID_START}, got '
            f'{combined["Open Time"].iloc[0]}.'
        )
    if train_rows != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f'Expected {EXPECTED_TRAIN_ROWS} Train rows, got {train_rows}.'
        )
    if test_rows != TEST_ROWS:
        raise RuntimeError(f'Expected {TEST_ROWS} Test rows, got {test_rows}.')

    # Fit scaling on all 32,444 Train rows. Test remains strictly excluded.
    price_scaler = FeatureStandardizer.fit(
        combined.loc[train_mask, price_cols]
    )
    tech_scaler = FeatureStandardizer.fit(
        combined.loc[train_mask, tech_cols]
    )
    combined.loc[:, price_cols] = price_scaler.transform(combined[price_cols])
    combined.loc[:, tech_cols] = tech_scaler.transform(combined[tech_cols])

    scaler_table = pd.concat(
        [
            price_scaler.to_frame('price'),
            tech_scaler.to_frame('technical_indicator'),
        ],
        ignore_index=True,
    )
    scaler_table.to_csv(DATA / 'feature_scaler.csv', index=False)

    for split_name, mask in [('train', train_mask), ('test', test_mask)]:
        c = combined.loc[mask].reset_index(drop=True)
        r = aligned_raw.loc[mask].reset_index(drop=True)
        c[ ['Open Time'] ].to_csv(DATA / f'timestamps_{split_name}.csv', index=False)
        c[price_cols].to_csv(DATA / f'pct_change_output_{split_name}.csv', index=False)
        c[tech_cols].to_csv(DATA / f'ta_test_{split_name}.csv', index=False)
        r.to_csv(DATA / f'merged_output_{split_name}.csv', index=False)
        # Explicit sanity checks before writing is considered successful.
        if not c['Open Time'].equals(r['Open Time']):
            raise RuntimeError(f'{split_name}: timestamp mismatch')

    print('Feature preparation complete.')
    print(f'Total valid rows : {len(combined):,}')
    print(f'Train rows       : {train_rows:,}')
    print(f'Test rows        : {test_rows:,}')
    print(f'Price features   : {len(price_cols)}')
    print(f'TA features      : {len(tech_cols)}')
    print(f'Train period     : {combined.loc[train_mask, "Open Time"].iloc[0]} -> {combined.loc[train_mask, "Open Time"].iloc[-1]}')
    print(f'Test period      : {combined.loc[test_mask, "Open Time"].iloc[0]} -> {combined.loc[test_mask, "Open Time"].iloc[-1]}')
    print('Scaler fit data  : Train only (Test excluded)')
    print(f'Scaler metadata  : {DATA / "feature_scaler.csv"}')
    print('Alignment check  : PASS')


if __name__ == '__main__':
    main()
