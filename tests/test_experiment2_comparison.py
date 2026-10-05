"""Experiment 2 新論文表格的欄位與計算測試。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_experiment2 import build_summary_table  # noqa: E402


class Experiment2ComparisonTests(unittest.TestCase):
    def test_summary_table_matches_paper_columns_and_formulas(self):
        timestamps = pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC")

        def frame(values, sharpe):
            return pd.DataFrame({
                "timestamp": timestamps,
                "portfolio_value": values,
                "expanding_sharpe_ratio": [np.nan, np.nan, 1.0, sharpe],
            })

        curves = {
            "Proposed Method": frame([100.0, 120.0, 108.0, 118.0], 2.6),
            "A2C": frame([100.0, 112.0, 105.0, 110.0], 1.8),
            "DQN": frame([100.0, 108.0, 90.0, 105.0], 1.5),
            "Buy and Hold": frame([100.0, 106.0, 98.0, 104.0], 1.7),
        }

        table = build_summary_table(curves)
        self.assertEqual(list(table.columns), [
            "Method",
            "Final PV",
            "Improve",
            "MDD",
            "Sharpe Ratio",
        ])
        self.assertEqual(list(table["Method"]), list(curves))

        proposed = table.loc[table["Method"] == "Proposed Method"].iloc[0]
        self.assertAlmostEqual(proposed["Final PV"], 118.0)
        self.assertAlmostEqual(proposed["Improve"], 118.0 / 104.0)
        self.assertAlmostEqual(proposed["MDD"], 108.0 / 120.0 - 1.0)
        self.assertAlmostEqual(proposed["Sharpe Ratio"], 2.6)

        buy_hold = table.loc[table["Method"] == "Buy and Hold"].iloc[0]
        self.assertAlmostEqual(buy_hold["Improve"], 1.0)


if __name__ == "__main__":
    unittest.main()
