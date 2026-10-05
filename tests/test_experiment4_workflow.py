"""Experiment 4 方法、步數標籤與架構公平性測試。"""

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config_4h as cfg  # noqa: E402
from compare_experiment4 import METHOD_LABELS, result_files  # noqa: E402
from train_common import model_kwargs  # noqa: E402
from src.original_mfn_extractor import OriginalTwoViewMFN  # noqa: E402


class Experiment4WorkflowTests(unittest.TestCase):
    def test_four_methods_and_dynamic_step_tag(self):
        files = result_files(300_000)
        self.assertEqual(set(files), set(METHOD_LABELS.values()))
        for name, filename in files.items():
            if name != METHOD_LABELS["buy_hold"]:
                self.assertIn("_300k_", filename)

    def test_original_mfn_uses_shared_a2c_settings(self):
        kwargs = model_kwargs("original_mfn")
        self.assertEqual(kwargs["learning_rate"], cfg.LEARNING_RATE)
        self.assertEqual(kwargs["n_steps"], cfg.A2C_N_STEPS)
        self.assertEqual(kwargs["gamma"], cfg.GAMMA)
        policy = kwargs["policy_kwargs"]
        self.assertIs(
            policy["features_extractor_class"], OriginalTwoViewMFN
        )
        self.assertEqual(
            policy["features_extractor_kwargs"]["lstm_hidden"], 64
        )
        self.assertEqual(
            policy["features_extractor_kwargs"]["memory_dim"], 128
        )


if __name__ == "__main__":
    unittest.main()
