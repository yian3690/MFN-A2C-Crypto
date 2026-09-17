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
from src.feature_schema import (
    CRYPTO_ASSETS,
    INDICATOR_DIM,
    PRICE_DIM,
    RELATIVE_STRENGTH_BARS,
    RELATIVE_STRENGTH_NAME,
)
from src.experiment_periods import (
    DATA_START,
    EXPECTED_DEVELOPMENT_ROWS,
    EXPECTED_TOTAL_VALID_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALID_START,
    TEST_END_EXCLUSIVE,
    TEST_ROWS,
    TEST_START,
    VALIDATION_ROWS,
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

    crypto_closes = pd.DataFrame(index=raw.index)
    for i, asset in enumerate(ASSETS):
        crypto_closes[asset] = pd.to_numeric(raw[f'Close{i}'], errors='coerce')

    # 方案A：以設定期間的過去報酬減去四種加密貨幣的
    # 同期平均報酬。這是橫截面相對強勢，只使用t及t以前的價格，
    # 不會讀取下一期或Test未來資料。
    rs_reference = crypto_closes.shift(RELATIVE_STRENGTH_BARS)
    # 原始資料最前段還沒有完整lookback歷史。
    # 為維持論文33,524筆有效資料，該小段使用「截至當時可取得的
    # 最長歷史」（第一根Close）作參考；滿lookback後改用固定期間。
    # 這個fallback只向後看，不會以未來值補資料。
    rs_reference = rs_reference.fillna(crypto_closes.iloc[0])
    momentum = crypto_closes / rs_reference - 1.0
    relative_strength = momentum.sub(momentum.mean(axis=1), axis=0)

    for i, asset in enumerate(ASSETS):
        c = pd.to_numeric(raw[f'Close{i}'], errors='coerce')

        # Thesis equation 4.2: one close-to-close price relative per asset.
        # At observation timestamp t this uses only Close[t] and Close[t-1].
        price[f'{asset}_price_relative'] = c / c.shift(1)

        sma = ta.sma(c, length=20)
        ema = ta.ema(c, length=20)
        rsi = ta.rsi(c, length=14)
        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if (
            any(value is None for value in (sma, ema, rsi, macd))
            or 'MACD_12_26_9' not in macd.columns
        ):
            raise RuntimeError(f'Technical-indicator calculation failed for {asset}')

        # 恢復較穩定的level＋Train-only z-score版本：先保留指標原始水準，
        # 等完成Train/Test切分後，再只用Train統計量統一標準化。
        tech[f'{asset}_SMA20'] = sma
        tech[f'{asset}_EMA20'] = ema
        # 沿用封存原碼的DIF／MACD line；不使用九期signal line。
        tech[f'{asset}_MACD'] = macd['MACD_12_26_9']
        tech[f'{asset}_RSI14'] = rsi
        tech[f'{asset}_{RELATIVE_STRENGTH_NAME}'] = relative_strength[asset]

    # USDT視為無風險資產。下列中性常數經Train-only z-score後皆為0，
    # 因此維持5項資產的一致欄位，不會向模型提供虛假USDT趨勢。
    price['USDT_price_relative'] = 1.0
    tech['USDT_SMA20'] = 1.0
    tech['USDT_EMA20'] = 1.0
    tech['USDT_MACD'] = 0.0
    tech['USDT_RSI14'] = 50.0
    # USDT不參與四種加密貨幣的橫截面排名，設為中性值；經z-score後仍為0。
    tech[f'USDT_{RELATIVE_STRENGTH_NAME}'] = 0.0

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

    test_mask = (
        (combined['Open Time'] >= TEST_START)
        & (combined['Open Time'] < TEST_END_EXCLUSIVE)
    )
    development_positions = np.flatnonzero(
        (combined['Open Time'] < TEST_START).to_numpy()
    )
    if len(development_positions) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError(
            f'Expected {EXPECTED_DEVELOPMENT_ROWS} pre-Test rows, got '
            f'{len(development_positions)}.'
        )
    validation_positions = development_positions[-VALIDATION_ROWS:]
    train_positions = development_positions[:-VALIDATION_ROWS]
    train_mask = np.zeros(len(combined), dtype=bool)
    validation_mask = np.zeros(len(combined), dtype=bool)
    train_mask[train_positions] = True
    validation_mask[validation_positions] = True

    train_rows = int(train_mask.sum())
    validation_rows = int(validation_mask.sum())
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
    if validation_rows != VALIDATION_ROWS:
        raise RuntimeError(
            f'Expected {VALIDATION_ROWS} Validation rows, got '
            f'{validation_rows}.'
        )
    if test_rows != TEST_ROWS:
        raise RuntimeError(f'Expected {TEST_ROWS} Test rows, got {test_rows}.')

    # Stage 1只用Train擬合，供Train/Validation選擇訓練步數。
    train_price_scaler = FeatureStandardizer.fit(
        combined.loc[train_mask, price_cols]
    )
    train_tech_scaler = FeatureStandardizer.fit(
        combined.loc[train_mask, tech_cols]
    )
    stage1_scaled = combined.copy()
    stage1_scaled.loc[:, price_cols] = train_price_scaler.transform(
        combined[price_cols]
    )
    stage1_scaled.loc[:, tech_cols] = train_tech_scaler.transform(
        combined[tech_cols]
    )

    # Stage 2可用Train＋Validation重擬Scaler，但Test仍完全排除。
    development_mask = train_mask | validation_mask
    development_price_scaler = FeatureStandardizer.fit(
        combined.loc[development_mask, price_cols]
    )
    development_tech_scaler = FeatureStandardizer.fit(
        combined.loc[development_mask, tech_cols]
    )
    final_scaled = combined.copy()
    final_scaled.loc[:, price_cols] = development_price_scaler.transform(
        combined[price_cols]
    )
    final_scaled.loc[:, tech_cols] = development_tech_scaler.transform(
        combined[tech_cols]
    )

    train_scaler_table = pd.concat(
        [
            train_price_scaler.to_frame('price'),
            train_tech_scaler.to_frame('technical_indicator'),
        ],
        ignore_index=True,
    )
    development_scaler_table = pd.concat(
        [
            development_price_scaler.to_frame('price'),
            development_tech_scaler.to_frame('technical_indicator'),
        ],
        ignore_index=True,
    )
    train_scaler_table.to_csv(DATA / 'feature_scaler_train.csv', index=False)
    development_scaler_table.to_csv(
        DATA / 'feature_scaler_development.csv',
        index=False,
    )
    # 本輪直接使用Validation最佳checkpoint評估Test，不做Stage 2；
    # 因此正式feature_scaler必須與Train模型一致，只能使用Train統計量。
    train_scaler_table.to_csv(DATA / 'feature_scaler.csv', index=False)

    for split_name, mask, scaled_source in [
        ('train', train_mask, stage1_scaled),
        ('validation', validation_mask, stage1_scaled),
        ('development', development_mask, final_scaled),
        ('test', test_mask, stage1_scaled),
    ]:
        c = scaled_source.loc[mask].reset_index(drop=True)
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
    print(f'Validation rows  : {validation_rows:,}')
    print(f'Test rows        : {test_rows:,}')
    print(f'Price features   : {len(price_cols)}')
    print(f'TA features      : {len(tech_cols)}')
    print(f'Train period     : {combined.loc[train_mask, "Open Time"].iloc[0]} -> {combined.loc[train_mask, "Open Time"].iloc[-1]}')
    print(f'Validation period: {combined.loc[validation_mask, "Open Time"].iloc[0]} -> {combined.loc[validation_mask, "Open Time"].iloc[-1]}')
    print(f'Test period      : {combined.loc[test_mask, "Open Time"].iloc[0]} -> {combined.loc[test_mask, "Open Time"].iloc[-1]}')
    print(f'Feature levels   : SMA / EMA / MACD(DIF) / RSI / {RELATIVE_STRENGTH_NAME}')
    print(
        f'{RELATIVE_STRENGTH_NAME} lookback : '
        f'{RELATIVE_STRENGTH_BARS} bars (14 days at 2H)'
    )
    print('Stage 1 scaling  : Train-only z-score')
    print('Test scaling     : Train-only z-score (Validation/Test excluded)')
    print(f'Train scaler     : {DATA / "feature_scaler_train.csv"}')
    print(f'Development scaler: {DATA / "feature_scaler_development.csv"}')
    print('Alignment check  : PASS')


if __name__ == '__main__':
    main()
