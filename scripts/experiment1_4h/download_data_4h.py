"""下載 2018-01-01 至 2025-09-01 的 Binance Spot 4H K 線。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_4h import DATA, DATA_END_INCLUSIVE, DATA_START

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "LTCUSDT", "BNBUSDT")
INTERVAL = "4h"
INTERVAL_MS = 4 * 60 * 60 * 1000
LIMIT = 1000
KLINE_COLUMNS = [
    "Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time",
    "Quote Asset Volume", "Number of Trades", "Taker Buy Base Asset Volume",
    "Taker Buy Quote Asset Volume", "Ignore",
]


def to_ms(value) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def request_rows(symbol: str, start_ms: int, end_ms: int) -> list:
    params = {
        "symbol": symbol, "interval": INTERVAL, "startTime": start_ms,
        "endTime": end_ms, "limit": LIMIT,
    }
    for attempt in range(5):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list):
                raise RuntimeError(f"Binance 回傳格式異常：{rows}")
            return rows
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    return []


def download_symbol(symbol: str) -> pd.DataFrame:
    cursor = to_ms(DATA_START)
    end_ms = to_ms(DATA_END_INCLUSIVE)
    rows: list = []
    while cursor <= end_ms:
        batch = request_rows(symbol, cursor, end_ms)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + INTERVAL_MS
        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol} 時間戳沒有向前推進。")
        cursor = next_cursor
        print(f"{symbol}: {len(rows):,} rows")
        if len(batch) < LIMIT:
            break
        time.sleep(0.15)

    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if frame.empty:
        raise RuntimeError(f"{symbol} 沒有下載到資料。")
    frame["Open Time"] = pd.to_datetime(frame["Open Time"], unit="ms", utc=True)
    frame["Close Time"] = pd.to_datetime(frame["Close Time"], unit="ms", utc=True)
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates("Open Time").sort_values("Open Time")
    frame = frame[frame["Open Time"] <= pd.Timestamp(DATA_END_INCLUSIVE)]
    frame.to_csv(DATA / f"{symbol}.csv", index=False)
    return frame


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    downloaded = {symbol: download_symbol(symbol) for symbol in SYMBOLS}
    merged = None
    for index, symbol in enumerate(SYMBOLS):
        part = downloaded[symbol][["Open Time", "Open", "High", "Low", "Close"]].copy()
        part = part.rename(columns={name: f"{name}{index}" for name in ("Open", "High", "Low", "Close")})
        merged = part if merged is None else merged.merge(part, on="Open Time", how="inner")
    merged = merged.sort_values("Open Time").reset_index(drop=True)
    output = DATA / "merged_output.csv"
    merged.to_csv(output, index=False)
    print(f"Saved {output}: {len(merged):,} rows")
    print(f"Period: {merged['Open Time'].iloc[0]} -> {merged['Open Time'].iloc[-1]}")


if __name__ == "__main__":
    main()
