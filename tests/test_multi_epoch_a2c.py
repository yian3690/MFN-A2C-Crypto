"""Tests for the thesis-configured repeated A2C updates."""

import unittest
from unittest.mock import MagicMock, patch

from stable_baselines3 import A2C

from src.multi_epoch_a2c import MultiEpochA2C


class MultiEpochA2CTests(unittest.TestCase):
    def test_train_repeats_parent_update(self):
        algorithm = object.__new__(MultiEpochA2C)
        algorithm.update_epochs = 18
        algorithm._logger = MagicMock()

        with patch.object(A2C, "train") as parent_train:
            MultiEpochA2C.train(algorithm)

        self.assertEqual(parent_train.call_count, 18)
        algorithm._logger.record.assert_called_once_with(
            "train/update_epochs",
            18,
        )

    def test_update_epochs_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            MultiEpochA2C(
                "MlpPolicy",
                None,
                update_epochs=0,
            )


if __name__ == "__main__":
    unittest.main()
