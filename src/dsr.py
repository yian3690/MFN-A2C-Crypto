"""Single source of truth for Differential Sharpe Ratio calculations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_ETA = 0.005
DEFAULT_WARMUP_STEPS = 5
DEFAULT_EPSILON = 1e-12
PAPER_FORMULA = "paper_legacy"
CANONICAL_FORMULA = "canonical"
LEGACY_EXPANDING_FORMULA = "legacy_expanding"
DEFAULT_FORMULA = PAPER_FORMULA


@dataclass
class DSRTracker:
    """Stateful Differential Sharpe Ratio calculator.

    EWMA paper/canonical modes and the archived expanding-mean mode share the
    same five-step warm-up. The archived mode stores prior returns because its
    A and B moments are recomputed from the full episode history.
    """

    eta: float = DEFAULT_ETA
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    epsilon: float = DEFAULT_EPSILON
    formula: str = DEFAULT_FORMULA
    first_moment: float = field(init=False, default=0.0)
    second_moment: float = field(init=False, default=0.0)
    step_count: int = field(init=False, default=0)
    return_history: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < self.eta <= 1.0:
            raise ValueError("eta must be in the interval (0, 1].")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.formula not in {
            PAPER_FORMULA,
            CANONICAL_FORMULA,
            LEGACY_EXPANDING_FORMULA,
        }:
            raise ValueError(
                "formula must be one of "
                f"'{PAPER_FORMULA}', '{CANONICAL_FORMULA}', or "
                f"'{LEGACY_EXPANDING_FORMULA}'."
            )

    def reset(self) -> None:
        """Reset moments and the warm-up counter for a new episode."""
        self.first_moment = 0.0
        self.second_moment = 0.0
        self.step_count = 0
        self.return_history = []

    def update(self, portfolio_return: float) -> float:
        """Update EWMA moments and return the configured DSR increment."""
        value = float(portfolio_return)

        if self.formula == LEGACY_EXPANDING_FORMULA:
            return self._update_legacy_expanding(value)

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

    def _update_legacy_expanding(self, value: float) -> float:
        """重現學長封存環境的 expanding-mean DSR step。

        封存原碼以「目前步驟以前」的全部投資組合報酬計算A與B，
        再把當期報酬視為innovation，最後將Dt乘上eta。這不同於
        論文文字的EWMA遞迴，因此保留為獨立消融模式。
        """
        current_step = self.step_count
        if current_step < self.warmup_steps or not self.return_history:
            dsr = 0.0
        else:
            history = np.asarray(self.return_history, dtype=np.float64)
            old_first = float(history.mean())
            old_second = float(np.square(history).mean())
            variance = old_second - old_first**2
            if variance <= self.epsilon:
                # 封存原碼沒有數值防護；這裡只避免0除與NaN傳入RL。
                dsr = 0.0
            else:
                delta_first = value - old_first
                delta_second = value**2 - old_second
                differential = (
                    old_second * delta_first
                    - 0.5 * old_first * delta_second
                ) / variance**1.5
                dsr = float(self.eta * differential)

        self.return_history.append(value)
        history_with_current = np.asarray(self.return_history, dtype=np.float64)
        self.first_moment = float(history_with_current.mean())
        self.second_moment = float(np.square(history_with_current).mean())
        self.step_count += 1
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
