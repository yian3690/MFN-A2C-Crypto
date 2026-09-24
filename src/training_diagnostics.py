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
    """Periodically persist rollout diagnostics to CSV and TensorBoard.

    ``record_every_rollouts`` only controls this read-only observer.  It does
    not skip environment steps, rollout collection, optimizer updates,
    Validation, or checkpoint callbacks.
    """

    def __init__(
        self,
        output_path: str | Path,
        verbose: int = 0,
        record_every_rollouts: int = 1,
    ):
        if (
            not isinstance(record_every_rollouts, int)
            or record_every_rollouts < 1
        ):
            raise ValueError("record_every_rollouts must be a positive integer.")
        super().__init__(verbose=verbose)
        self.output_path = Path(output_path)
        self.record_every_rollouts = record_every_rollouts
        self._values: dict[str, list] = defaultdict(list)
        self._pending: dict[str, float] | None = None
        self._header_written = False
        self._rollout_count = 0
        self._record_current_rollout = False

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if not self._record_current_rollout:
            return True
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
            if "log_portfolio_return" in info:
                self._values["log_portfolio_return"].append(
                    info["log_portfolio_return"]
                )
            if "scaled_DSR_reward" in info:
                self._values["scaled_dsr_component"].append(
                    info["scaled_DSR_reward"]
                )
            if "scaled_return_reward" in info:
                self._values["scaled_return_component"].append(
                    info["scaled_return_reward"]
                )
        return True

    def _distribution_metrics(
        self,
        sampled_weights: np.ndarray | None = None,
    ) -> dict[str, float]:
        """比較訓練抽樣動作與deterministic投資比例。"""
        buffer = self.model.rollout_buffer
        observations = buffer.observations.reshape(
            (-1, *self.model.observation_space.shape)
        )
        obs_tensor, _ = self.model.policy.obs_to_tensor(observations)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            torch_distribution = distribution.distribution
            if hasattr(torch_distribution, "concentration"):
                concentration = torch_distribution.concentration
                policy_weights = concentration / concentration.sum(
                    dim=-1, keepdim=True
                )
                alpha = concentration.detach().cpu().numpy()
            else:
                # SB3 Gaussian policy的deterministic action是mean logits；
                # 先依Box範圍裁切，再套用與環境相同的Softmax。
                logits = torch_distribution.mean
                low = torch.as_tensor(
                    self.model.action_space.low,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                high = torch.as_tensor(
                    self.model.action_space.high,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                clipped_logits = torch.clamp(logits, min=low, max=high)
                policy_weights = torch.softmax(clipped_logits, dim=-1)
                alpha = None

        weights = policy_weights.detach().cpu().numpy()
        safe_weights = np.clip(weights, 1e-12, 1.0)
        deterministic_entropy = (
            -np.sum(safe_weights * np.log(safe_weights), axis=1)
            / np.log(len(ASSET_NAMES))
        )
        metrics = {
            "policy_weight_temporal_std_mean": float(weights.std(axis=0).mean()),
            "policy_equal_weight_l1_mean": float(
                (0.5 * np.abs(weights - 0.2).sum(axis=1)).mean()
            ),
            "deterministic_weight_entropy_mean": float(
                deterministic_entropy.mean()
            ),
            "deterministic_weight_turnover_mean": float(
                0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=1).mean()
            ) if len(weights) > 1 else 0.0,
        }
        if sampled_weights is not None and len(sampled_weights) == len(weights):
            sampled = np.asarray(sampled_weights, dtype=float)
            metrics["sampled_vs_deterministic_weight_l1_mean"] = float(
                (0.5 * np.abs(sampled - weights).sum(axis=1)).mean()
            )
            metrics["sampled_weight_turnover_mean"] = float(
                0.5 * np.abs(np.diff(sampled, axis=0)).sum(axis=1).mean()
            ) if len(sampled) > 1 else 0.0
        if alpha is not None:
            metrics.update(
                dirichlet_alpha_mean=float(alpha.mean()),
                dirichlet_alpha_min=float(alpha.min()),
                dirichlet_alpha_max=float(alpha.max()),
                dirichlet_total_concentration_mean=float(
                    alpha.sum(axis=1).mean()
                ),
                dirichlet_alpha_below_one_fraction=float((alpha < 1.0).mean()),
            )
        else:
            raw_logits = torch_distribution.mean.detach().cpu().numpy()
            stddev = torch_distribution.stddev.detach().cpu().numpy()
            if stddev.ndim == 1:
                stddev = np.broadcast_to(stddev, raw_logits.shape)
            metrics.update(
                gaussian_logit_abs_max=float(np.abs(raw_logits).max()),
                gaussian_logit_clip_fraction=float(
                    (np.abs(raw_logits) >= 5.0 - 1e-6).mean()
                ),
                gaussian_std_mean=float(stddev.mean()),
                gaussian_std_min=float(stddev.min()),
                gaussian_std_max=float(stddev.max()),
            )
            for index, asset in enumerate(ASSET_NAMES):
                asset_std = float(stddev[:, index].mean())
                metrics[f"gaussian_std_{asset.lower()}"] = asset_std
                metrics[f"gaussian_log_std_{asset.lower()}"] = float(
                    np.log(max(asset_std, 1e-12))
                )
        return metrics

    def _critic_metrics(self) -> dict[str, float]:
        """記錄Critic目標、預測與Advantage的尺度及吻合程度。"""
        buffer = self.model.rollout_buffer
        values = np.asarray(buffer.values, dtype=float).reshape(-1)
        returns = np.asarray(buffer.returns, dtype=float).reshape(-1)
        advantages = np.asarray(buffer.advantages, dtype=float).reshape(-1)
        metrics = {
            "value_prediction_mean": float(values.mean()),
            "value_prediction_std": float(values.std()),
            "return_target_mean": float(returns.mean()),
            "return_target_std": float(returns.std()),
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            "advantage_abs_mean": float(np.abs(advantages).mean()),
            "advantage_p01": float(np.quantile(advantages, 0.01)),
            "advantage_p99": float(np.quantile(advantages, 0.99)),
            "value_target_rmse": float(np.sqrt(np.mean((returns - values) ** 2))),
        }
        if values.std() > 0 and returns.std() > 0:
            metrics["value_return_correlation"] = float(
                np.corrcoef(values, returns)[0, 1]
            )
        else:
            metrics["value_return_correlation"] = np.nan
        return metrics

    def _on_rollout_end(self) -> None:
        if not self._record_current_rollout:
            self._values = defaultdict(list)
            self._rollout_count += 1
            return

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

        for key in (
            "portfolio_return",
            "log_portfolio_return",
            "dsr",
            "scaled_dsr_component",
            "scaled_return_component",
            "portfolio_value",
            "turnover",
            "dsr_variance",
        ):
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

        dsr_component = np.asarray(
            self._values.get("scaled_dsr_component", []), dtype=float
        )
        return_component = np.asarray(
            self._values.get("scaled_return_component", []), dtype=float
        )
        if dsr_component.size and dsr_component.size == return_component.size:
            dsr_abs_sum = float(np.abs(dsr_component).sum())
            return_abs_sum = float(np.abs(return_component).sum())
            component_abs_sum = dsr_abs_sum + return_abs_sum
            active = (~np.isclose(dsr_component, 0.0)) & (
                ~np.isclose(return_component, 0.0)
            )
            row["dsr_component_abs_sum"] = dsr_abs_sum
            row["return_component_abs_sum"] = return_abs_sum
            row["dsr_component_abs_fraction"] = (
                dsr_abs_sum / component_abs_sum if component_abs_sum else np.nan
            )
            row["return_component_abs_fraction"] = (
                return_abs_sum / component_abs_sum if component_abs_sum else np.nan
            )
            row["reward_sign_conflict_fraction"] = (
                float(
                    (
                        np.sign(dsr_component[active])
                        != np.sign(return_component[active])
                    ).mean()
                )
                if active.any()
                else 0.0
            )
            row["reward_component_correlation"] = (
                float(np.corrcoef(dsr_component, return_component)[0, 1])
                if dsr_component.std() > 0 and return_component.std() > 0
                else np.nan
            )

        try:
            row.update(self._distribution_metrics(weights))
            row.update(self._critic_metrics())
        except (AttributeError, RuntimeError, ValueError):
            # Keep training alive if a non-Dirichlet policy reuses the callback.
            pass

        self._pending = row
        self._values = defaultdict(list)
        self._rollout_count += 1

    def _parameter_metrics(self) -> dict[str, float]:
        extractor = self.model.policy.features_extractor
        parameter_sq = 0.0
        gradient_sq = 0.0
        gradient_max = 0.0
        gradient_elements = 0
        near_zero_gradients = 0
        parameter_elements = 0
        for parameter in extractor.parameters():
            parameter_elements += parameter.numel()
            parameter_sq += float(parameter.detach().square().sum().cpu())
            if parameter.grad is not None:
                gradient = parameter.grad.detach()
                gradient_sq += float(gradient.square().sum().cpu())
                gradient_max = max(gradient_max, float(gradient.abs().max().cpu()))
                gradient_elements += gradient.numel()
                near_zero_gradients += int((gradient.abs() < 1e-12).sum().cpu())
        if parameter_elements == 0:
            return {
                "mfn_parameter_norm": np.nan,
                "mfn_gradient_norm": np.nan,
                "mfn_gradient_abs_max": np.nan,
                "mfn_near_zero_gradient_fraction": np.nan,
            }
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
        self._record_current_rollout = (
            self._rollout_count % self.record_every_rollouts == 0
        )

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
        "gaussian_logit_clip_fraction_recent": median(
            "gaussian_logit_clip_fraction"
        ),
        "turnover_recent": median("turnover_mean"),
        "gaussian_std_recent": median("gaussian_std_mean"),
        "sampled_deterministic_gap_recent": median(
            "sampled_vs_deterministic_weight_l1_mean"
        ),
        "value_return_correlation_recent": median(
            "value_return_correlation"
        ),
        "reward_sign_conflict_recent": median(
            "reward_sign_conflict_fraction"
        ),
    }
    # SB3 MlpPolicy的FlattenExtractor沒有可訓練參數；舊baseline CSV曾以0
    # 記錄MFN梯度，不能解讀為credit assignment失敗。
    if "mfn_a2c_" not in Path(path).name.lower():
        summary["mfn_gradient_norm_recent"] = np.nan

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
    clip_fraction = summary["gaussian_logit_clip_fraction_recent"]
    if np.isfinite(clip_fraction) and clip_fraction > 0.10:
        warnings.append("Gaussian action logits frequently hit the Box limit; Softmax allocation may be saturated.")
    if np.isfinite(summary["turnover_recent"]) and summary["turnover_recent"] > 0.25:
        warnings.append("Turnover is high; no-cost training may reward noisy reallocation.")
    if (
        np.isfinite(summary["sampled_deterministic_gap_recent"])
        and summary["sampled_deterministic_gap_recent"] > 0.20
    ):
        warnings.append(
            "Sampled training allocations differ strongly from deterministic "
            "allocations; Gaussian exploration may obscure the deployed policy."
        )
    if (
        np.isfinite(summary["value_return_correlation_recent"])
        and summary["value_return_correlation_recent"] < 0.20
    ):
        warnings.append(
            "Critic value predictions have weak correlation with return targets."
        )
    if (
        np.isfinite(summary["reward_sign_conflict_recent"])
        and summary["reward_sign_conflict_recent"] > 0.40
    ):
        warnings.append(
            "DSR and log-return reward components frequently have opposite signs."
        )
    if not warnings:
        warnings.append("No single threshold failure was detected; compare trends across seeds and checkpoints.")
    return summary, warnings


def latest_diagnostics(directory: str | Path, pattern: str) -> Path | None:
    """Return the newest matching diagnostics CSV, if one exists."""
    candidates = list(Path(directory).glob(pattern))
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def print_training_diagnostics_summary(
    path: str | Path,
    summary: dict[str, float],
    warnings: list[str],
) -> None:
    """將最重要的探索、Critic與reward衝突診斷輸出到終端機。"""
    print("TRAINING DIAGNOSTICS")
    print(f"Source CSV                    : {path}")
    print(f"Recorded diagnostic rows      : {int(summary['rollouts'])}")
    fields = (
        ("Recent reward std", "reward_std_recent", ".6g"),
        ("Recent |reward| p99", "reward_abs_p99_recent", ".6g"),
        ("Recent explained variance", "explained_variance_recent", ".4f"),
        ("Value/return correlation", "value_return_correlation_recent", ".4f"),
        ("Gaussian std", "gaussian_std_recent", ".4f"),
        (
            "Sampled/deterministic gap",
            "sampled_deterministic_gap_recent",
            ".2%",
        ),
        ("Reward sign conflict", "reward_sign_conflict_recent", ".2%"),
        ("Recent allocation entropy", "allocation_entropy_recent", ".4f"),
        ("Recent deterministic variation", "policy_weight_variation_recent", ".4f"),
        ("Recent sampled turnover", "turnover_recent", ".2%"),
    )
    for label, key, number_format in fields:
        value = summary.get(key, np.nan)
        text = format(value, number_format) if np.isfinite(value) else "n/a"
        print(f"{label:<30}: {text}")
    gradient = summary.get("mfn_gradient_norm_recent", np.nan)
    if np.isfinite(gradient):
        print(f"{'Recent MFN gradient norm':<30}: {gradient:.6g}")
    print("Possible causes:")
    for warning in warnings:
        print(f"- {warning}")
