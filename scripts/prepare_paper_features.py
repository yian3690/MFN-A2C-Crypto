from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
MERGED = DATA / 'merged_output.csv'
TEST_ROWS = 1080
ASSETS = ['BTC','ETH','LTC','BNB']


def main():
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

    price = pd.DataFrame({'Open Time': raw['Open Time']})
    tech = pd.DataFrame({'Open Time': raw['Open Time']})

    for i, asset in enumerate(ASSETS):
        o = pd.to_numeric(raw[f'Open{i}'], errors='coerce')
        h = pd.to_numeric(raw[f'High{i}'], errors='coerce')
        l = pd.to_numeric(raw[f'Low{i}'], errors='coerce')
        c = pd.to_numeric(raw[f'Close{i}'], errors='coerce')

        for name, s in [('Open',o),('High',h),('Low',l),('Close',c)]:
            price[f'{asset}_{name}_ret'] = s.pct_change() * 100.0

        sma = ta.sma(c, length=20)
        ema = ta.ema(c, length=20)
        rsi = ta.rsi(c, length=14)
        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if any(v is None for v in (sma, ema, rsi, macd)) or 'MACD_12_26_9' not in macd.columns:
            raise RuntimeError(f'Technical-indicator calculation failed for {asset}')
        tech[f'{asset}_SMA20'] = sma.pct_change() * 100.0
        tech[f'{asset}_EMA20'] = ema.pct_change() * 100.0
        tech[f'{asset}_RSI14'] = (rsi - 50.0) * 0.1
        tech[f'{asset}_MACD'] = macd['MACD_12_26_9'].pct_change() * 100.0

    # THE FIX: join all modalities and raw rows by the SAME timestamp before dropna.
    combined = price.merge(tech, on='Open Time', how='inner', validate='one_to_one')
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    price_cols = [c for c in combined.columns if c.endswith('_ret')]
    tech_cols = [c for c in combined.columns if c != 'Open Time' and c not in price_cols]
    if len(price_cols) != 16 or len(tech_cols) != 16:
        raise RuntimeError(f'Expected 16+16 features, got {len(price_cols)}+{len(tech_cols)}')
    if len(combined) <= TEST_ROWS:
        raise RuntimeError(f'Only {len(combined)} valid rows remain; cannot reserve {TEST_ROWS} test rows.')

    # THE FIX: recover raw OHLC using exact timestamps, never tail slicing.
    raw_indexed = raw.set_index('Open Time')
    aligned_raw = raw_indexed.loc[combined['Open Time']].reset_index()
    if not aligned_raw['Open Time'].equals(combined['Open Time']):
        raise RuntimeError('FINAL ALIGNMENT FAILED: raw and feature timestamps differ.')

    split = len(combined) - TEST_ROWS
    for split_name, sl in [('train', slice(0, split)), ('test', slice(split, None))]:
        c = combined.iloc[sl].reset_index(drop=True)
        r = aligned_raw.iloc[sl].reset_index(drop=True)
        c[ ['Open Time'] ].to_csv(DATA / f'timestamps_{split_name}.csv', index=False)
        c[price_cols].to_csv(DATA / f'pct_change_output_{split_name}.csv', index=False)
        c[tech_cols].to_csv(DATA / f'ta_test_{split_name}.csv', index=False)
        r.to_csv(DATA / f'merged_output_{split_name}.csv', index=False)
        # Explicit sanity checks before writing is considered successful.
        if not c['Open Time'].equals(r['Open Time']):
            raise RuntimeError(f'{split_name}: timestamp mismatch')

    print('Feature preparation complete.')
    print(f'Total valid rows : {len(combined):,}')
    print(f'Train rows       : {split:,}')
    print(f'Test rows        : {TEST_ROWS:,}')
    print(f'Price features   : {len(price_cols)}')
    print(f'TA features      : {len(tech_cols)}')
    print(f'Train period     : {combined["Open Time"].iloc[0]} -> {combined["Open Time"].iloc[split-1]}')
    print(f'Test period      : {combined["Open Time"].iloc[split]} -> {combined["Open Time"].iloc[-1]}')
    print('Alignment check  : PASS')


if __name__ == '__main__':
    main()
