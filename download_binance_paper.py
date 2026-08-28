"""
Download the Binance Spot 4-hour K-line data used by the paper.

Paper setting:
- Symbols: BTC, ETH, LTC, BNB against USDT
- Interval: 4 hours
- Start: 2018-01-01
- End: 2025-09-01
- The existing project uses four crypto columns + USDT cash.

This script uses Binance's public market-data REST endpoint.
No API key is required for public K-line data.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.binance.com/api/v3/klines"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "BNBUSDT"]
INTERVAL = "4h"

# Match the date range in the uploaded binance.py.
START_DATE = "2018-01-01 00:00:00+00:00"
END_DATE = "2025-09-01 00:00:00+00:00"

OUT_DIR = Path("data")
MERGED_FILE = Path("merged_output.csv")

LIMIT = 1000
REQUEST_SLEEP = 0.15
MAX_RETRIES = 5


KLINE_COLUMNS = [
    "Open Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close Time",
    "Quote Asset Volume",
    "Number of Trades",
    "Taker Buy Base Asset Volume",
    "Taker Buy Quote Asset Volume",
    "Ignore",
]


def to_ms(date_string: str) -> int:
    return int(pd.Timestamp(date_string).timestamp() * 1000)


def request_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected Binance response: {data}")

            return data

        except Exception as exc:
            print(f"[{symbol}] request failed ({attempt}/{MAX_RETRIES}): {exc}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** (attempt - 1))

    return []


def download_symbol(symbol: str) -> pd.DataFrame:
    start_ms = to_ms(START_DATE)
    end_ms = to_ms(END_DATE)

    all_rows = []
    interval_ms = 4 * 60 * 60 * 1000

    print(f"\nDownloading {symbol}: {START_DATE} -> {END_DATE}")

    while start_ms < end_ms:
        rows = request_klines(symbol, start_ms, end_ms)

        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = int(rows[-1][0])
        next_start = last_open_time + interval_ms

        if next_start <= start_ms:
            raise RuntimeError(
                f"{symbol}: Binance returned non-progressing timestamps."
            )

        start_ms = next_start

        print(
            f"  fetched={len(all_rows):>6} "
            f"last={pd.to_datetime(last_open_time, unit='ms', utc=True)}"
        )

        if len(rows) < LIMIT:
            break

        time.sleep(REQUEST_SLEEP)

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)

    if df.empty:
        raise RuntimeError(
            f"No data returned for {symbol}. "
            "Check Binance API availability and symbol listing."
        )

    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms", utc=True)

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote Asset Volume",
        "Taker Buy Base Asset Volume",
        "Taker Buy Quote Asset Volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Number of Trades"] = pd.to_numeric(
        df["Number of Trades"], errors="coerce"
    ).astype("Int64")

    df = df.drop_duplicates("Open Time").sort_values("Open Time")
    df = df[
        (df["Open Time"] >= pd.Timestamp(START_DATE))
        & (df["Open Time"] < pd.Timestamp(END_DATE))
    ].reset_index(drop=True)

    symbol_file = OUT_DIR / f"{symbol}.csv"
    df.to_csv(symbol_file, index=False)

    print(
        f"Saved {symbol_file} | rows={len(df):,} | "
        f"{df['Open Time'].min()} -> {df['Open Time'].max()}"
    )

    return df


def build_merged_output(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the format expected by the existing project:
    Open0, High0, Low0, Close0,
    Open1, High1, Low1, Close1, ...

    Order follows the paper:
    BTC, ETH, LTC, BNB.
    USDT is handled by the trading environment as cash with price=1.
    """

    merged = None

    for idx, symbol in enumerate(SYMBOLS):
        df = dataframes[symbol][
            ["Open Time", "Open", "High", "Low", "Close"]
        ].copy()

        rename_map = {
            "Open": f"Open{idx}",
            "High": f"High{idx}",
            "Low": f"Low{idx}",
            "Close": f"Close{idx}",
        }
        df = df.rename(columns=rename_map)

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="Open Time", how="inner")

    if merged is None or merged.empty:
        raise RuntimeError("Merged dataset is empty.")

    merged = merged.sort_values("Open Time").reset_index(drop=True)

    # The old project does not use timestamp columns directly,
    # but keeping it makes auditing and debugging much easier.
    merged.to_csv(MERGED_FILE, index=False)

    print(
        f"\nSaved {MERGED_FILE} | rows={len(merged):,} | "
        f"columns={len(merged.columns)}"
    )
    print(merged.head(3).to_string(index=False))
    print("\nMissing values:")
    print(merged.isna().sum())

    return merged


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for symbol in SYMBOLS:
        downloaded[symbol] = download_symbol(symbol)

    build_merged_output(downloaded)

    print("\nDone.")
    print("Files:")
    for symbol in SYMBOLS:
        print(f"  {OUT_DIR / (symbol + '.csv')}")
    print(f"  {MERGED_FILE}")


if __name__ == "__main__":
    main()
