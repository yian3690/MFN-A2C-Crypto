"""Leakage-safe feature standardization shared by data preparation and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureStandardizer:
    """Per-column z-score parameters fitted on training data only."""

    mean: pd.Series
    scale: pd.Series
    fit_rows: int

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "FeatureStandardizer":
        """Fit population mean/std on Train without using held-out Test rows."""
        numeric = frame.apply(pd.to_numeric, errors="raise").astype(np.float64)
        mean = numeric.mean(axis=0)
        scale = numeric.std(axis=0, ddof=0)
        scale = scale.mask(scale.abs() < np.finfo(np.float64).eps, 1.0)
        return cls(mean=mean, scale=scale, fit_rows=len(numeric))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply stored training statistics while preserving columns/index."""
        if list(frame.columns) != list(self.mean.index):
            raise ValueError("Feature columns do not match the fitted standardizer.")

        numeric = frame.apply(pd.to_numeric, errors="raise").astype(np.float64)
        transformed = (numeric - self.mean) / self.scale
        if not np.isfinite(transformed.to_numpy()).all():
            raise ValueError("Standardized features contain NaN or infinity.")
        return transformed

    def to_frame(self, feature_group: str) -> pd.DataFrame:
        """Return serializable scaler metadata for reproducibility."""
        return pd.DataFrame(
            {
                "feature_group": feature_group,
                "feature": self.mean.index,
                "mean": self.mean.to_numpy(),
                "scale": self.scale.to_numpy(),
                "fit_rows": self.fit_rows,
            }
        )
