"""目前唯一4H實驗的時間範圍與Train/Test列數。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

UTC = timezone.utc
BAR_INTERVAL = timedelta(hours=4)
BARS_PER_DAY = 6
PERIODS_PER_YEAR = BARS_PER_DAY * 365
LOOKBACK = 20

DATA_START = datetime(2018, 1, 1, tzinfo=UTC)
DATA_END_INCLUSIVE = datetime(2025, 9, 1, tzinfo=UTC)
TEST_END_EXCLUSIVE = DATA_END_INCLUSIVE + BAR_INTERVAL
TEST_ROWS = 1080
TEST_START = TEST_END_EXCLUSIVE - TEST_ROWS * BAR_INTERVAL

# 技術指標warm-up後，目前有效特徵為15,680筆Train＋1,080筆Test。
EXPECTED_TRAIN_ROWS = 15_680
EXPECTED_TOTAL_VALID_ROWS = 16_760
EXPECTED_DEVELOPMENT_ROWS = EXPECTED_TRAIN_ROWS
EXPECTED_VALID_START = datetime(2018, 1, 5, 4, tzinfo=UTC)

if EXPECTED_TRAIN_ROWS + TEST_ROWS != EXPECTED_TOTAL_VALID_ROWS:
    raise RuntimeError("4H Train/Test列數總和不一致。")
if TEST_START + TEST_ROWS * BAR_INTERVAL != TEST_END_EXCLUSIVE:
    raise RuntimeError("Test period is not aligned to the 4-hour grid.")
