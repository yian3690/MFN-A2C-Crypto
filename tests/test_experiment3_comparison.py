"""Experiment 3 新論文表格的欄位與計算測試。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_experiment3 import (  # noqa: E402
    METHOD_LABELS,
    build_summary_table,
)


class Experiment3ComparisonTests(unittest.TestCase):
    def test_summary_table_uses_a2c_return_as_improvement_baseline(self):
        timestamps = pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC")

        def frame(values, sharpe):
            return pd.DataFrame({
                "timestamp": timestamps,
                "portfolio_value": values,
                "expanding_sharpe_ratio": [np.nan, np.nan, 1.0, sharpe],
            })

        curves = {
            METHOD_LABELS["proposed_dsr"]: frame(
                [100.0, 120.0, 108.0, 118.0], 2.6
            ),
            METHOD_LABELS["proposed_return"]: frame(
                [100.0, 110.0, 104.0, 108.0], 1.8
            ),
            METHOD_LABELS["a2c_dsr"]: frame(
                [100.0, 112.0, 105.0, 110.0], 2.0
            ),
            METHOD_LABELS["a2c_return"]: frame(
                [100.0, 106.0, 98.0, 104.0], 1.7
            ),
        }

        table = build_summary_table(curves)
        self.assertEqual(list(table.columns), [
            "Method",
            "Final PV",
            "Improve",
            "MDD",
            "Sharpe Ratio",
        ])
        self.assertEqual(list(table["Method"]), list(METHOD_LABELS.values()))

        proposed = table.loc[
            table["Method"] == METHOD_LABELS["proposed_dsr"]
        ].iloc[0]
        self.assertAlmostEqual(proposed["Final PV"], 118.0)
        self.assertAlmostEqual(proposed["Improve"], 118.0 / 104.0)
        self.assertAlmostEqual(proposed["MDD"], 108.0 / 120.0 - 1.0)
        self.assertAlmostEqual(proposed["Sharpe Ratio"], 2.6)

        baseline = table.loc[
            table["Method"] == METHOD_LABELS["a2c_return"]
        ].iloc[0]
        self.assertAlmostEqual(baseline["Improve"], 1.0)


if __name__ == "__main__":
    unittest.main()
