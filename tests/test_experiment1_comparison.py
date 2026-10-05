"""Experiment 1新Table II欄位與計算測試。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_experiment1 import build_summary_table  # noqa: E402


class Experiment1ComparisonTests(unittest.TestCase):
    def test_summary_table_matches_new_columns_and_formulas(self):
        timestamps = pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC")

        def frame(values, sharpe):
            return pd.DataFrame({
                "timestamp": timestamps,
                "portfolio_value": values,
                "expanding_sharpe_ratio": [np.nan, np.nan, 1.0, sharpe],
            })

        curves = {
            "Proposed Method": frame([100.0, 110.0, 105.0, 120.0], 2.0),
            "A2C": frame([100.0, 105.0, 101.0, 110.0], 1.5),
            "A2C w/o ti": frame([100.0, 102.0, 99.0, 105.0], 1.0),
            "Buy and Hold": frame([100.0, 104.0, 98.0, 108.0], 1.2),
        }
        table = build_summary_table(curves)
        self.assertEqual(list(table.columns), [
            "Method",
            "Peak PV",
            "Peak Improve",
            "Final PV",
            "Final Improve",
            "Total Return",
            "Max Drawdown",
            "Final Expanding Sharpe",
        ])
        proposed = table.loc[table["Method"] == "Proposed Method"].iloc[0]
        self.assertAlmostEqual(proposed["Peak Improve"], 120.0 / 108.0)
        self.assertAlmostEqual(proposed["Final Improve"], 120.0 / 108.0)
        self.assertAlmostEqual(proposed["Total Return"], 0.20)
        self.assertAlmostEqual(proposed["Final Expanding Sharpe"], 2.0)


if __name__ == "__main__":
    unittest.main()
