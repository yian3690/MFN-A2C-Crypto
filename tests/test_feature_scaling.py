"""Tests for leakage-safe feature standardization."""

import unittest
import numpy as np
import pandas as pd

from src.feature_scaling import FeatureStandardizer


class FeatureStandardizerTestCase(unittest.TestCase):
    def test_fit_rows_are_zero_mean_and_unit_variance(self):
        fit = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
        scaler = FeatureStandardizer.fit(fit)
        transformed = scaler.transform(fit)

        np.testing.assert_allclose(
            transformed.mean().to_numpy(), 0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            transformed.std(ddof=0).to_numpy(), 1.0, atol=1e-12
        )
        self.assertEqual(scaler.fit_rows, 3)

    def test_validation_values_do_not_change_fitted_statistics(self):
        fit = pd.DataFrame({"feature": [0.0, 1.0, 2.0]})
        validation = pd.DataFrame({"feature": [1_000_000.0]})
        scaler = FeatureStandardizer.fit(fit)

        transformed_validation = scaler.transform(validation)

        self.assertEqual(scaler.mean["feature"], 1.0)
        self.assertGreater(transformed_validation.iloc[0, 0], 1_000.0)

    def test_constant_column_remains_finite(self):
        fit = pd.DataFrame({"constant": [5.0, 5.0, 5.0]})
        scaler = FeatureStandardizer.fit(fit)

        self.assertEqual(scaler.scale["constant"], 1.0)
        self.assertTrue(np.isfinite(scaler.transform(fit).to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
