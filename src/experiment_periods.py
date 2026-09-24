"""主2H實驗的時間範圍與Train/Test列數單一來源。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


UTC = timezone.utc
BAR_INTERVAL = timedelta(hours=2)
BARS_PER_DAY = 12
PERIODS_PER_YEAR = BARS_PER_DAY * 365
# 20根2小時K線，等於40小時歷史觀察窗。
LOOKBACK = 20

DATA_START = datetime(2018, 1, 1, tzinfo=UTC)
DATA_END_INCLUSIVE = datetime(2025, 9, 1, tzinfo=UTC)
TEST_END_EXCLUSIVE = DATA_END_INCLUSIVE + BAR_INTERVAL

# 論文描述只切分Train與Test：32,444＋1,080＝33,524筆有效資料。
TEST_ROWS = 1080
TEST_START = TEST_END_EXCLUSIVE - TEST_ROWS * BAR_INTERVAL
EXPECTED_TOTAL_VALID_ROWS = 33524
EXPECTED_TRAIN_ROWS = 32444
# 保留舊名稱作唯讀相容別名；不再代表獨立Development階段。
EXPECTED_DEVELOPMENT_ROWS = EXPECTED_TRAIN_ROWS

# 公開Binance封存資料有少數歷史缺口，不能僅用列數反推第一時間戳。
EXPECTED_VALID_START = datetime(2018, 1, 3, 4, tzinfo=UTC)

if EXPECTED_TRAIN_ROWS + TEST_ROWS != EXPECTED_TOTAL_VALID_ROWS:
    raise RuntimeError("Train/Test列數總和不等於有效資料總數。")
if TEST_START + TEST_ROWS * BAR_INTERVAL != TEST_END_EXCLUSIVE:
    raise RuntimeError("Test period is not aligned to the 2-hour grid.")
