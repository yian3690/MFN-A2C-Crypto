"""Tests for MFN checkpoint discovery used by ``--resume``."""

import tempfile
import unittest
from pathlib import Path

from scripts.train_mfn_a2c import (
    CHECKPOINT_PREFIX,
    checkpoint_timesteps,
    find_latest_checkpoint,
)


class TrainMFNResumeTests(unittest.TestCase):
    def test_checkpoint_timesteps_rejects_incompatible_models(self):
        valid = Path(f"{CHECKPOINT_PREFIX}_200000_steps.zip")
        incompatible = Path("MFN_A2C_900000_steps.zip")

        self.assertEqual(checkpoint_timesteps(valid), 200_000)
        self.assertIsNone(checkpoint_timesteps(incompatible))

    def test_find_latest_uses_greatest_absolute_timestep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / f"{CHECKPOINT_PREFIX}_100000_steps.zip"
            newer = root / f"{CHECKPOINT_PREFIX}_200000_steps.zip"
            unrelated = root / "MFN_A2C_999999_steps.zip"
            for path in (newer, unrelated, older):
                path.touch()

            self.assertEqual(find_latest_checkpoint(root), newer)

    def test_find_latest_returns_none_without_compatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(find_latest_checkpoint(directory))


if __name__ == "__main__":
    unittest.main()
