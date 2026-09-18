"""Single source of truth for the chronological experiment periods."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


UTC = timezone.utc
BAR_INTERVAL = timedelta(hours=2)
BARS_PER_DAY = 12
PERIODS_PER_YEAR = BARS_PER_DAY * 365
# 20根2小時K線，等於40小時歷史觀察窗。
LOOKBACK = 20

# Original 2-hour collection period. The thesis' 33,550 raw-row count matches
# Binance when the candle starting at 2025-09-01 00:00 UTC is included.
DATA_START = datetime(2018, 1, 1, tzinfo=UTC)
DATA_END_INCLUSIVE = datetime(2025, 9, 1, tzinfo=UTC)
TEST_END_EXCLUSIVE = DATA_END_INCLUSIVE + BAR_INTERVAL

# 固定保留最後1,080筆作Test；再由原Train尾端取540筆（45天）作Validation。
# Test永遠不參與checkpoint選擇，避免實驗層級的資料洩漏。
TEST_ROWS = 1080
VALIDATION_ROWS = 540
TEST_START = TEST_END_EXCLUSIVE - TEST_ROWS * BAR_INTERVAL

# 論文原始切分為32,444 development rows＋1,080 Test；加入45天Validation後，
# 實際Train為31,904，Validation為540，Test仍為1,080。
EXPECTED_TOTAL_VALID_ROWS = 33524
EXPECTED_DEVELOPMENT_ROWS = 32444
EXPECTED_TRAIN_ROWS = EXPECTED_DEVELOPMENT_ROWS - VALIDATION_ROWS
# The Binance archive contains historical gaps, so row count cannot be
# converted into the first timestamp by assuming a perfectly continuous grid.
EXPECTED_VALID_START = datetime(2018, 1, 3, 4, tzinfo=UTC)

if TEST_START + TEST_ROWS * BAR_INTERVAL != TEST_END_EXCLUSIVE:
    raise RuntimeError("Test period is not aligned to the 2-hour grid.")
