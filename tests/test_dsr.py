"""Regression tests for the project's single shared DSR implementation."""

import unittest

import numpy as np

from src.dsr import (
    CANONICAL_FORMULA,
    DEFAULT_WARMUP_STEPS,
    DSRTracker,
    LEGACY_EXPANDING_FORMULA,
    PAPER_FORMULA,
    calculate_dsr_series,
)


class DSRTestCase(unittest.TestCase):
    def setUp(self):
        self.returns = np.array(
            [0.01, -0.02, 0.005, 0.003, -0.004, 0.012, -0.006, 0.008],
            dtype=np.float64,
        )

    def test_default_uses_five_step_warmup(self):
        self.assertEqual(DEFAULT_WARMUP_STEPS, 5)
        values = calculate_dsr_series(self.returns)
        np.testing.assert_array_equal(values[:5], np.zeros(5))
        self.assertNotEqual(values[5], 0.0)

    def test_explicit_no_warmup_remains_available(self):
        values = calculate_dsr_series(self.returns, warmup_steps=0)
        self.assertNotEqual(values[1], 0.0)

    def test_tracker_and_series_are_identical(self):
        tracker = DSRTracker()
        online = np.asarray(
            [tracker.update(value) for value in self.returns],
            dtype=np.float64,
        )
        offline = calculate_dsr_series(self.returns)
        np.testing.assert_array_equal(online, offline)

    def test_reset_restarts_moments_and_counter(self):
        tracker = DSRTracker()
        first = [tracker.update(value) for value in self.returns]
        tracker.reset()
        second = [tracker.update(value) for value in self.returns]
        np.testing.assert_array_equal(first, second)

    def test_default_uses_paper_ewma_moment_changes(self):
        eta = 0.005
        tracker = DSRTracker(eta=eta, warmup_steps=0)
        tracker.update(0.01)

        old_a = tracker.first_moment
        old_b = tracker.second_moment
        value = -0.02
        new_a = old_a + eta * (value - old_a)
        new_b = old_b + eta * (value**2 - old_b)
        expected = (
            old_b * (new_a - old_a)
            - 0.5 * old_a * (new_b - old_b)
        ) / (old_b - old_a**2) ** 1.5

        actual = tracker.update(value)
        self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(
            tracker.first_moment,
            old_a + eta * (value - old_a),
        )
        self.assertAlmostEqual(
            tracker.second_moment,
            old_b + eta * (value**2 - old_b),
        )

    def test_paper_reward_is_eta_scaled_canonical_reward(self):
        eta = 0.005
        paper = DSRTracker(
            eta=eta,
            warmup_steps=0,
            formula=PAPER_FORMULA,
        )
        canonical = DSRTracker(
            eta=eta,
            warmup_steps=0,
            formula=CANONICAL_FORMULA,
        )
        paper.update(0.01)
        canonical.update(0.01)
        value = -0.02

        paper_reward = paper.update(value)
        canonical_reward = canonical.update(value)
        self.assertAlmostEqual(paper_reward, eta * canonical_reward)

    def test_invalid_formula_is_rejected(self):
        with self.assertRaises(ValueError):
            DSRTracker(formula="unknown")

    def test_legacy_expanding_matches_archived_equation(self):
        eta = 0.005
        tracker = DSRTracker(
            eta=eta,
            warmup_steps=5,
            formula=LEGACY_EXPANDING_FORMULA,
        )
        values = [tracker.update(value) for value in self.returns]
        np.testing.assert_array_equal(values[:5], np.zeros(5))

        history = self.returns[:5]
        current = self.returns[5]
        mean = float(history.mean())
        second = float(np.square(history).mean())
        expected = eta * (
            second * (current - mean)
            - 0.5 * mean * (current**2 - second)
        ) / (second - mean**2) ** 1.5
        self.assertAlmostEqual(values[5], expected)


if __name__ == "__main__":
    unittest.main()
