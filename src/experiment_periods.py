"""Single source of truth for the chronological experiment periods."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


UTC = timezone.utc
BAR_INTERVAL = timedelta(hours=2)
BARS_PER_DAY = 12
PERIODS_PER_YEAR = BARS_PER_DAY * 365
LOOKBACK = 20

# Original 2-hour collection period. The thesis' 33,550 raw-row count matches
# Binance when the candle starting at 2025-09-01 00:00 UTC is included.
DATA_START = datetime(2018, 1, 1, tzinfo=UTC)
DATA_END_INCLUSIVE = datetime(2025, 9, 1, tzinfo=UTC)
TEST_END_EXCLUSIVE = DATA_END_INCLUSIVE + BAR_INTERVAL

# Paper-reproduction split: all rows before the final 1,080-row Test period
# are used for training. There is no Validation split or checkpoint selection.
TEST_ROWS = 1080
TEST_START = TEST_END_EXCLUSIVE - TEST_ROWS * BAR_INTERVAL

# The thesis reports 33,524 valid rows: 32,444 Train + 1,080 Test.
EXPECTED_TOTAL_VALID_ROWS = 33524
EXPECTED_TRAIN_ROWS = 32444
# The Binance archive contains historical gaps, so row count cannot be
# converted into the first timestamp by assuming a perfectly continuous grid.
EXPECTED_VALID_START = datetime(2018, 1, 3, 4, tzinfo=UTC)

if TEST_START + TEST_ROWS * BAR_INTERVAL != TEST_END_EXCLUSIVE:
    raise RuntimeError("Test period is not aligned to the 2-hour grid.")
