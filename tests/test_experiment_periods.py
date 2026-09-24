<<<<<<< Updated upstream
"""Regression tests for the fixed chronological experiment periods."""
=======
"""4H Train/Test固定時間範圍的回歸測試。"""
>>>>>>> Stashed changes

import unittest
from datetime import datetime, timezone

import pandas as pd

from src.evaluation_metrics import (
    add_test_timestamps,
    validate_saved_result_period,
)
from src.experiment_periods import (
<<<<<<< Updated upstream
    BAR_INTERVAL,
    PERIODS_PER_YEAR,
    EXPECTED_TOTAL_VALID_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALID_START,
    TEST_END_EXCLUSIVE,
    TEST_ROWS,
    TEST_START,
=======
    BAR_INTERVAL, EXPECTED_TOTAL_VALID_ROWS, EXPECTED_TRAIN_ROWS,
    EXPECTED_VALID_START, LOOKBACK, PERIODS_PER_YEAR,
    TEST_END_EXCLUSIVE, TEST_ROWS, TEST_START,
>>>>>>> Stashed changes
)


class ExperimentPeriodsTestCase(unittest.TestCase):
    def test_periods_are_strictly_chronological(self):
        self.assertLess(TEST_START, TEST_END_EXCLUSIVE)

    def test_expected_row_counts(self):
<<<<<<< Updated upstream
        self.assertEqual(TEST_ROWS, 1080)
        self.assertEqual(EXPECTED_TOTAL_VALID_ROWS, 33524)
        self.assertEqual(EXPECTED_TRAIN_ROWS, 32444)
        self.assertEqual(PERIODS_PER_YEAR, 12 * 365)
        self.assertEqual(
            EXPECTED_VALID_START,
            datetime(2018, 1, 3, 4, tzinfo=timezone.utc),
        )
        self.assertEqual(
            TEST_START + TEST_ROWS * BAR_INTERVAL,
            TEST_END_EXCLUSIVE,
        )

    def test_saved_result_requires_the_fixed_test_timeline(self):
        raw = pd.DataFrame(
            {
                "Open Time": pd.date_range(
                    TEST_START,
                    periods=TEST_ROWS,
                    freq="2h",
                )
            }
        )
        result = pd.DataFrame(
            {"portfolio_value": [10_000.0] * (TEST_ROWS - 20)}
        )
=======
        self.assertEqual(EXPECTED_TOTAL_VALID_ROWS, 16_760)
        self.assertEqual(EXPECTED_TRAIN_ROWS, 15_680)
        self.assertEqual(TEST_ROWS, 1_080)
        self.assertEqual(PERIODS_PER_YEAR, 6 * 365)
        self.assertEqual(EXPECTED_VALID_START,
                         datetime(2018, 1, 5, 4, tzinfo=timezone.utc))
        self.assertEqual(TEST_START + TEST_ROWS * BAR_INTERVAL,
                         TEST_END_EXCLUSIVE)

    def test_saved_result_requires_fixed_4h_timeline(self):
        raw = pd.DataFrame({
            "Open Time": pd.date_range(TEST_START, periods=TEST_ROWS, freq="4h")
        })
        result = pd.DataFrame({
            "portfolio_value": [10_000.0] * (TEST_ROWS - LOOKBACK)
        })
>>>>>>> Stashed changes
        stamped = add_test_timestamps(result, raw)

        validate_saved_result_period(stamped)
        self.assertEqual(len(stamped), TEST_ROWS - 20)

    def test_stale_result_without_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no timestamp"):
            validate_saved_result_period(pd.DataFrame({"portfolio_value": [10_000.0]}))


if __name__ == "__main__":
    unittest.main()
