"""Validate row-by-row alignment among timestamps, raw prices, and engineered features."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

for split in ('train','test'):
    ts = pd.read_csv(DATA / f'timestamps_{split}.csv')
    raw = pd.read_csv(DATA / f'merged_output_{split}.csv')
    price = pd.read_csv(DATA / f'pct_change_output_{split}.csv')
    tech = pd.read_csv(DATA / f'ta_test_{split}.csv')
    ts['Open Time'] = pd.to_datetime(ts['Open Time'], utc=True)
    raw['Open Time'] = pd.to_datetime(raw['Open Time'], utc=True)
    ok = (len(ts)==len(raw)==len(price)==len(tech) and ts['Open Time'].equals(raw['Open Time']) and price.notna().all().all() and tech.notna().all().all())
    print(f'{split.upper()}: {"PASS" if ok else "FAIL"} | rows={len(ts):,} | price={price.shape} | tech={tech.shape}')
    if not ok: raise SystemExit(1)
print('All alignment checks passed.')
