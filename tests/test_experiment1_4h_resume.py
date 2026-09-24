"""4H五方法跨目標步數續訓與診斷頻率測試。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config_4h as cfg  # noqa: E402
from train_common import (  # noqa: E402
    METHOD_LABELS,
    checkpoint_prefix,
    checkpoint_timesteps,
    find_latest_checkpoint,
    normalized_model_name,
    parse_args,
)


class Experiment1FourHourResumeTests(unittest.TestCase):
    def test_all_methods_use_five_rollout_diagnostics(self) -> None:
        self.assertEqual(cfg.DIAGNOSTICS_EVERY_ROLLOUTS, 5)
        self.assertEqual(set(METHOD_LABELS), set(cfg.MODEL_NAMES))

    def test_resume_prefix_excludes_target_step_tag(self) -> None:
        for method in METHOD_LABELS:
            with self.subTest(method=method):
                normalized = normalized_model_name(method)
                self.assertNotIn(f"_{cfg.STEP_TAG.upper()}_", normalized)
                self.assertTrue(checkpoint_prefix(method).endswith("_RESUME"))

    def test_finds_new_and_legacy_compatible_checkpoints(self) -> None:
        method = "without_ti"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / f"{cfg.MODEL_NAMES[method].upper()}_300000_steps.zip"
            stable = root / f"{checkpoint_prefix(method)}_500000_steps.zip"
            incompatible = root / "A2C_WITHOUT_TI_4H_VALSELECT_600K_600000_steps.zip"
            for path in (legacy, stable, incompatible):
                path.touch()

            self.assertEqual(checkpoint_timesteps(legacy), 300_000)
            self.assertEqual(
                find_latest_checkpoint(root, method, max_timesteps=400_000),
                legacy,
            )
            self.assertEqual(
                find_latest_checkpoint(root, method, max_timesteps=600_000),
                stable,
            )

    def test_resume_flag_is_explicit(self) -> None:
        self.assertFalse(parse_args("a2c", []).resume)
        self.assertTrue(parse_args("a2c", ["--resume"]).resume)


if __name__ == "__main__":
    unittest.main()
