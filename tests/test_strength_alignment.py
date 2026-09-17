"""Tests for relative-strength allocation diagnostics."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation_metrics import (
    add_strength_alignment_columns,
    summarize_strength_alignment,
)


class StrengthAlignmentTests(unittest.TestCase):
    def test_known_momentum_follower_has_positive_rank_correlation(self):
        rows = 100
        assets = ("btc", "eth", "ltc", "bnb", "usdt")
        result = pd.DataFrame()
        weights = (0.10, 0.15, 0.20, 0.25, 0.30)
        returns = (-0.01, 0.00, 0.01, 0.02, 0.03)
        for asset, weight, asset_return in zip(assets, weights, returns):
            result[f"weight_{asset}"] = np.full(rows, weight)
            result[f"asset_return_{asset}"] = np.full(rows, asset_return)

        diagnosed = add_strength_alignment_columns(result)
        metrics = summarize_strength_alignment(diagnosed)

        self.assertAlmostEqual(metrics["6-step Rank Correlation"], 1.0)
        self.assertAlmostEqual(metrics["84-step Rank Correlation"], 1.0)
        self.assertAlmostEqual(metrics["12-step Trailing Winner Weight"], 0.30)
        self.assertAlmostEqual(metrics["12-step Trailing Loser Weight"], 0.10)
        self.assertAlmostEqual(metrics["Realized Winner Weight"], 0.30)
        self.assertAlmostEqual(metrics["Realized Loser Weight"], 0.10)

    def test_trailing_momentum_does_not_use_current_return(self):
        rows = 10
        assets = ("btc", "eth", "ltc", "bnb", "usdt")
        result = pd.DataFrame()
        for asset in assets:
            result[f"weight_{asset}"] = np.full(rows, 0.20)
            result[f"asset_return_{asset}"] = np.zeros(rows)
        result.loc[6, "asset_return_btc"] = 1.0

        diagnosed = add_strength_alignment_columns(result, horizons=(6,))

        # 第6列的動量只能看到第0至5列；第6列BTC暴漲要到第7列才可見。
        self.assertAlmostEqual(diagnosed.loc[6, "trailing_return_6_btc"], 0.0)
        self.assertAlmostEqual(diagnosed.loc[7, "trailing_return_6_btc"], 1.0)


if __name__ == "__main__":
    unittest.main()
