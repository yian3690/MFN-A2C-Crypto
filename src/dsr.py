"""Single source of truth for Differential Sharpe Ratio calculations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_ETA = 0.005
DEFAULT_WARMUP_STEPS = 5
DEFAULT_EPSILON = 1e-12
PAPER_FORMULA = "paper_legacy"
CANONICAL_FORMULA = "canonical"
DEFAULT_FORMULA = PAPER_FORMULA


@dataclass
class DSRTracker:
    """Stateful EWMA Differential Sharpe Ratio calculator.

    For the first five steps, the exponentially weighted moments are updated
    normally but the returned DSR increment is zero. This reproduces the
    warm-up used by the project's original experiment implementation and
    prevents unstable rewards while the variance estimate is being formed.
    """

    eta: float = DEFAULT_ETA
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    epsilon: float = DEFAULT_EPSILON
    formula: str = DEFAULT_FORMULA
    first_moment: float = field(init=False, default=0.0)
    second_moment: float = field(init=False, default=0.0)
    step_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not 0.0 < self.eta <= 1.0:
            raise ValueError("eta must be in the interval (0, 1].")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.formula not in {PAPER_FORMULA, CANONICAL_FORMULA}:
            raise ValueError(
                f"formula must be '{PAPER_FORMULA}' or '{CANONICAL_FORMULA}'."
            )

    def reset(self) -> None:
        """Reset moments and the warm-up counter for a new episode."""
        self.first_moment = 0.0
        self.second_moment = 0.0
        self.step_count = 0

    def update(self, portfolio_return: float) -> float:
        """Update EWMA moments and return the configured DSR increment."""
        value = float(portfolio_return)
        old_first = self.first_moment
        old_second = self.second_moment

        innovation_first = value - old_first
        innovation_second = value**2 - old_second
        new_first = old_first + self.eta * innovation_first
        new_second = old_second + self.eta * innovation_second

        if self.formula == PAPER_FORMULA:
            # Reproduce the archived project/paper implementation: the DSR
            # numerator uses the actual EWMA moment changes. This scales the
            # reward by eta relative to the canonical innovation definition.
            delta_first = new_first - old_first
            delta_second = new_second - old_second
        else:
            delta_first = innovation_first
            delta_second = innovation_second

        variance = old_second - old_first**2

        current_step = self.step_count
        self.step_count += 1
        if current_step < self.warmup_steps or variance <= self.epsilon:
            dsr = 0.0
        else:
            dsr = float(
                (old_second * delta_first - 0.5 * old_first * delta_second)
                / variance**1.5
            )

        self.first_moment = new_first
        self.second_moment = new_second
        return dsr


def calculate_dsr_series(
    returns,
    eta: float = DEFAULT_ETA,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    formula: str = DEFAULT_FORMULA,
) -> np.ndarray:
    """Calculate DSR increments by replaying returns through ``DSRTracker``."""
    tracker = DSRTracker(
        eta=eta,
        warmup_steps=warmup_steps,
        formula=formula,
    )
    return np.asarray(
        [tracker.update(value) for value in returns],
        dtype=np.float64,
    )
