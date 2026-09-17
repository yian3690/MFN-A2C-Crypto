"""學長原碼技術指標轉換的回歸測試。"""

import unittest

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.technical_indicators import (
    build_neutral_usdt_indicators,
    build_senior_original_indicators,
)


class SeniorTechnicalIndicatorTests(unittest.TestCase):
    def setUp(self):
        # 非等比數列可避免MACD長期固定而產生無意義的0/0百分比變化。
        x = np.arange(1.0, 121.0)
        self.close = pd.Series(100.0 + 0.2 * x + np.sin(x / 4.0))

    def test_matches_archived_transformations(self):
        actual = build_senior_original_indicators(self.close)
        macd = ta.macd(self.close, fast=12, slow=26, signal=9)

        expected = pd.DataFrame(
            {
                "SMA20": ta.sma(self.close, length=20).pct_change(
                    fill_method=None
                ) * 100.0,
                "EMA20": ta.ema(self.close, length=20).pct_change(
                    fill_method=None
                ) * 100.0,
                "MACD": macd["MACD_12_26_9"].pct_change(
                    fill_method=None
                ) * 100.0,
                "RSI14": (ta.rsi(self.close, length=14) - 50.0) * 0.1,
            }
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_usdt_columns_are_neutral(self):
        actual = build_neutral_usdt_indicators(self.close.index)
        self.assertEqual(
            list(actual.columns), ["SMA20", "EMA20", "MACD", "RSI14"]
        )
        np.testing.assert_array_equal(actual.to_numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()
