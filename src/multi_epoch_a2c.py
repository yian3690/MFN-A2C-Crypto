"""A2C variant that repeats optimization on each collected rollout."""

from __future__ import annotations

from stable_baselines3 import A2C


class MultiEpochA2C(A2C):
    """Run multiple A2C optimizer passes over the same rollout buffer.

    Stable-Baselines3 A2C normally performs one full-batch optimizer step per
    rollout. The thesis hyperparameter table reports 18 update epochs, so this
    subclass reuses each on-policy rollout for that many consecutive updates.
    No PPO clipping or other algorithm change is introduced.
    """

    def __init__(self, *args, update_epochs: int = 18, **kwargs):
        if not isinstance(update_epochs, int) or update_epochs < 1:
            raise ValueError("update_epochs must be a positive integer.")
        self.update_epochs = update_epochs
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        """Repeat the standard SB3 A2C update on the current rollout."""
        for _ in range(self.update_epochs):
            super().train()
        self.logger.record("train/update_epochs", self.update_epochs)
