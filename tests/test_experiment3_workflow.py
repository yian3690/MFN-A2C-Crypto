"""Experiment 3四方法架構一致性與動態輸出測試。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config_4h as cfg  # noqa: E402
from compare_experiment3 import METHOD_LABELS, _save_figure, result_files  # noqa: E402
from experiment3_common import _model_kwargs, _names  # noqa: E402
from train_common import model_kwargs as dsr_model_kwargs  # noqa: E402


class Experiment3WorkflowTests(unittest.TestCase):
    def test_four_result_files_follow_requested_step_budget(self):
        files = result_files(300_000)
        self.assertEqual(len(files), 4)
        self.assertEqual(set(files), set(METHOD_LABELS.values()))
        self.assertTrue(all("_300k_" in name for name in files.values()))

    def test_return_variants_reuse_their_dsr_architecture(self):
        for method in ("a2c", "dman_attention"):
            with self.subTest(method=method):
                self.assertEqual(_model_kwargs(method), dsr_model_kwargs(method))

    def test_return_names_are_distinct_and_dynamic(self):
        a2c_model, a2c_result = _names("a2c")
        proposed_model, proposed_result = _names("dman_attention")
        self.assertNotEqual(a2c_model, proposed_model)
        self.assertIn(cfg.STEP_TAG, a2c_model)
        self.assertIn(cfg.STEP_TAG, proposed_model)
        self.assertTrue(a2c_result.endswith("_results.csv"))
        self.assertTrue(proposed_result.endswith("_results.csv"))
        self.assertIn("reward_portfolio_return", a2c_model)
        self.assertIn("reward_portfolio_return", proposed_model)
        self.assertNotIn("paperdsr", a2c_model)
        self.assertNotIn("paperdsr", proposed_model)

    def test_figure_writer_creates_requested_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "experiment3" / "figure.png"
            output.parent.mkdir(parents=True)
            fig, ax = plt.subplots()
            ax.plot([0, 1], [10_000, 11_000])
            _save_figure(fig, output)
            plt.close(fig)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
