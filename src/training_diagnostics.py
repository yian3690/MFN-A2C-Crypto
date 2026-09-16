"""Rollout-level diagnostics for explaining weak portfolio-agent training."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback


ASSET_NAMES = ("BTC", "ETH", "LTC", "BNB", "USDT")


def _finite_scalar(value, default=np.nan) -> float:
    """Convert logger values to a finite scalar when possible."""
    try:
        scalar = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return float(default)
    return scalar if np.isfinite(scalar) else float(default)


class TrainingDiagnosticsCallback(BaseCallback):
    """Persist one diagnostic row per rollout to CSV and TensorBoard."""

    def __init__(self, output_path: str | Path, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.output_path = Path(output_path)
        self._values: dict[str, list] = defaultdict(list)
        self._pending: dict[str, float] | None = None
        self._header_written = False

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "portfolio_return" in info:
                self._values["portfolio_return"].append(info["portfolio_return"])
            if "DSR" in info:
                self._values["dsr"].append(info["DSR"])
            if "portfolio_value" in info:
                self._values["portfolio_value"].append(info["portfolio_value"])
            if "turnover" in info:
                self._values["turnover"].append(info["turnover"])
            if "weights" in info:
                self._values["weights"].append(np.asarray(info["weights"], dtype=float))
            if "dsr_variance" in info:
                self._values["dsr_variance"].append(info["dsr_variance"])
        return True

    def _distribution_metrics(self) -> dict[str, float]:
        """Measure Dirichlet concentrations and deterministic policy spread."""
        buffer = self.model.rollout_buffer
        observations = buffer.observations.reshape(
            (-1, *self.model.observation_space.shape)
        )
        if len(observations) > 256:
            indices = np.linspace(0, len(observations) - 1, 256, dtype=int)
            observations = observations[indices]

        obs_tensor, _ = self.model.policy.obs_to_tensor(observations)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            concentration = distribution.distribution.concentration
            policy_weights = concentration / concentration.sum(
                dim=-1, keepdim=True
            )

        alpha = concentration.detach().cpu().numpy()
        weights = policy_weights.detach().cpu().numpy()
        return {
            "dirichlet_alpha_mean": float(alpha.mean()),
            "dirichlet_alpha_min": float(alpha.min()),
            "dirichlet_alpha_max": float(alpha.max()),
            "dirichlet_total_concentration_mean": float(alpha.sum(axis=1).mean()),
            "dirichlet_alpha_below_one_fraction": float((alpha < 1.0).mean()),
            "policy_weight_temporal_std_mean": float(weights.std(axis=0).mean()),
            "policy_equal_weight_l1_mean": float(
                (0.5 * np.abs(weights - 0.2).sum(axis=1)).mean()
            ),
        }

    def _on_rollout_end(self) -> None:
        rewards = np.asarray(self.model.rollout_buffer.rewards, dtype=float).reshape(-1)
        abs_rewards = np.abs(rewards)
        row: dict[str, float] = {
            "timesteps": float(self.num_timesteps),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_min": float(rewards.min()),
            "reward_max": float(rewards.max()),
            "reward_abs_max": float(abs_rewards.max()),
            "reward_abs_p99": float(np.quantile(abs_rewards, 0.99)),
            "reward_zero_fraction": float(np.isclose(rewards, 0.0).mean()),
        }

        for key in ("portfolio_return", "dsr", "portfolio_value", "turnover", "dsr_variance"):
            values = np.asarray(self._values.get(key, []), dtype=float)
            if values.size:
                row[f"{key}_mean"] = float(values.mean())
                row[f"{key}_std"] = float(values.std())
                row[f"{key}_min"] = float(values.min())
                row[f"{key}_max"] = float(values.max())

        weights = np.asarray(self._values.get("weights", []), dtype=float)
        if weights.size:
            safe = np.clip(weights, 1e-12, 1.0)
            entropy = -np.sum(safe * np.log(safe), axis=1) / np.log(len(ASSET_NAMES))
            row["allocation_entropy_mean"] = float(entropy.mean())
            row["allocation_hhi_mean"] = float(np.square(weights).sum(axis=1).mean())
            row["allocation_equal_weight_l1_mean"] = float(
                (0.5 * np.abs(weights - 0.2).sum(axis=1)).mean()
            )
            for index, asset in enumerate(ASSET_NAMES):
                row[f"weight_{asset.lower()}_mean"] = float(weights[:, index].mean())
                row[f"weight_{asset.lower()}_max"] = float(weights[:, index].max())

        try:
            row.update(self._distribution_metrics())
        except (AttributeError, RuntimeError, ValueError):
            # Keep training alive if a non-Dirichlet policy reuses the callback.
            pass

        self._pending = row
        self._values = defaultdict(list)

    def _parameter_metrics(self) -> dict[str, float]:
        extractor = self.model.policy.features_extractor
        parameter_sq = 0.0
        gradient_sq = 0.0
        gradient_max = 0.0
        gradient_elements = 0
        near_zero_gradients = 0
        for parameter in extractor.parameters():
            parameter_sq += float(parameter.detach().square().sum().cpu())
            if parameter.grad is not None:
                gradient = parameter.grad.detach()
                gradient_sq += float(gradient.square().sum().cpu())
                gradient_max = max(gradient_max, float(gradient.abs().max().cpu()))
                gradient_elements += gradient.numel()
                near_zero_gradients += int((gradient.abs() < 1e-12).sum().cpu())
        return {
            "mfn_parameter_norm": parameter_sq**0.5,
            "mfn_gradient_norm": gradient_sq**0.5,
            "mfn_gradient_abs_max": gradient_max,
            "mfn_near_zero_gradient_fraction": (
                near_zero_gradients / gradient_elements if gradient_elements else np.nan
            ),
        }

    def _finalize_pending(self) -> None:
        if self._pending is None:
            return

        logger_values = self.logger.name_to_value
        for output_name, logger_name in {
            "policy_loss": "train/policy_loss",
            "value_loss": "train/value_loss",
            "entropy_loss": "train/entropy_loss",
            "explained_variance": "train/explained_variance",
            "learning_rate": "train/learning_rate",
            "n_updates": "train/n_updates",
        }.items():
            self._pending[output_name] = _finite_scalar(logger_values.get(logger_name))
        self._pending.update(self._parameter_metrics())

        frame = pd.DataFrame([self._pending])
        frame.to_csv(
            self.output_path,
            mode="a",
            header=not self._header_written,
            index=False,
        )
        self._header_written = True
        for key, value in self._pending.items():
            if key != "timesteps" and np.isfinite(value):
                self.logger.record(f"diagnostics/{key}", value)
        self._pending = None

    def _on_rollout_start(self) -> None:
        # This runs after the preceding rollout's optimizer update, allowing
        # its losses and resulting gradients to be attached to that rollout.
        self._finalize_pending()

    def _on_training_end(self) -> None:
        self._finalize_pending()


def diagnose_training_file(path: str | Path) -> tuple[dict[str, float], list[str]]:
    """Summarize a diagnostics CSV and return evidence-based warnings."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Training diagnostics file is empty.")
    tail = frame.tail(min(20, len(frame)))

    def median(column: str) -> float:
        if column not in tail:
            return float("nan")
        return float(pd.to_numeric(tail[column], errors="coerce").median())

    summary = {
        "rollouts": float(len(frame)),
        "final_timesteps": float(frame["timesteps"].iloc[-1]),
        "reward_std_recent": median("reward_std"),
        "reward_abs_p99_recent": median("reward_abs_p99"),
        "explained_variance_recent": median("explained_variance"),
        "mfn_gradient_norm_recent": median("mfn_gradient_norm"),
        "allocation_entropy_recent": median("allocation_entropy_mean"),
        "equal_weight_distance_recent": median("policy_equal_weight_l1_mean"),
        "policy_weight_variation_recent": median("policy_weight_temporal_std_mean"),
        "turnover_recent": median("turnover_mean"),
    }

    warnings: list[str] = []
    if np.isfinite(summary["reward_std_recent"]) and summary["reward_std_recent"] < 1e-4:
        warnings.append("DSR reward variation is extremely small; the policy signal may be too weak.")
    if np.isfinite(summary["reward_abs_p99_recent"]) and summary["reward_abs_p99_recent"] > 50:
        warnings.append("DSR reward has large spikes; early EWMA variance or reward instability may dominate learning.")
    if np.isfinite(summary["explained_variance_recent"]) and summary["explained_variance_recent"] < 0:
        warnings.append("Critic explained variance is negative; value learning is worse than a constant predictor.")
    if np.isfinite(summary["mfn_gradient_norm_recent"]) and summary["mfn_gradient_norm_recent"] < 1e-8:
        warnings.append("MFN gradients are nearly zero; credit is not reaching the feature extractor.")
    entropy = summary["allocation_entropy_recent"]
    equal_distance = summary["equal_weight_distance_recent"]
    variation = summary["policy_weight_variation_recent"]
    if np.isfinite(entropy) and np.isfinite(equal_distance) and entropy > 0.98 and equal_distance < 0.05:
        warnings.append("Policy remains close to equal weight; features/reward have not produced asset differentiation.")
    if np.isfinite(entropy) and entropy < 0.35:
        warnings.append("Portfolio policy is highly concentrated; possible action-distribution collapse.")
    if np.isfinite(variation) and variation < 0.005:
        warnings.append("Deterministic policy barely changes across states; the actor may be ignoring observations.")
    if np.isfinite(summary["turnover_recent"]) and summary["turnover_recent"] > 0.25:
        warnings.append("Turnover is high; no-cost training may reward noisy reallocation.")
    if not warnings:
        warnings.append("No single threshold failure was detected; compare trends across seeds and checkpoints.")
    return summary, warnings


def latest_diagnostics(directory: str | Path, pattern: str) -> Path | None:
    """Return the newest matching diagnostics CSV, if one exists."""
    candidates = list(Path(directory).glob(pattern))
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None

