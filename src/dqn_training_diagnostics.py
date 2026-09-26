"""DQN 專用的離策略訓練診斷，不假設存在 A2C rollout buffer。"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class DQNTrainingDiagnosticsCallback(BaseCallback):
    """每固定環境步數寫入 reward、投資組合、探索與 Q 值診斷。"""

    def __init__(self, output_path: str | Path, record_every_steps: int = 10_000):
        if not isinstance(record_every_steps, int) or record_every_steps < 1:
            raise ValueError("record_every_steps must be a positive integer.")
        super().__init__()
        self.output_path = Path(output_path)
        self.record_every_steps = record_every_steps
        self._values: dict[str, list] = defaultdict(list)
        self._next_record = record_every_steps
        self._header_written = False

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # resume 時 num_timesteps 不從 0 開始，下一筆對齊後續區間。
        self._next_record = (
            self.num_timesteps // self.record_every_steps + 1
        ) * self.record_every_steps

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
        self._values["reward"].extend(rewards.tolist())
        for info in self.locals.get("infos", []):
            for source, target in (
                ("portfolio_return", "portfolio_return"),
                ("DSR", "dsr"),
                ("turnover", "turnover"),
                ("portfolio_value", "portfolio_value"),
            ):
                if source in info:
                    self._values[target].append(float(info[source]))
            if "weights" in info:
                weights = np.asarray(info["weights"], dtype=float)
                safe = np.clip(weights, 1e-12, 1.0)
                self._values["allocation_entropy"].append(
                    float(-np.sum(safe * np.log(safe)) / np.log(len(weights)))
                )
        if self.num_timesteps >= self._next_record:
            self._write_row()
            self._values = defaultdict(list)
            while self._next_record <= self.num_timesteps:
                self._next_record += self.record_every_steps
        return True

    @staticmethod
    def _summary(values: list) -> tuple[float, float]:
        array = np.asarray(values, dtype=float)
        if not array.size:
            return np.nan, np.nan
        return float(array.mean()), float(array.std())

    def _q_metrics(self) -> dict[str, float]:
        buffer_size = int(self.model.replay_buffer.size())
        row = {"replay_buffer_size": float(buffer_size)}
        if buffer_size < 1:
            return row
        batch_size = min(256, buffer_size)
        samples = self.model.replay_buffer.sample(batch_size)
        with torch.no_grad():
            q_values = self.model.q_net(samples.observations)
            chosen = q_values.gather(1, samples.actions.long()).flatten()
        row.update(
            q_value_mean=float(q_values.mean().cpu()),
            q_value_std=float(q_values.std(unbiased=False).cpu()),
            q_value_abs_max=float(q_values.abs().max().cpu()),
            chosen_q_mean=float(chosen.mean().cpu()),
        )
        return row

    def _write_row(self) -> None:
        row: dict[str, float] = {
            "timesteps": float(self.num_timesteps),
            "exploration_rate": float(self.model.exploration_rate),
        }
        for key in (
            "reward", "portfolio_return", "dsr", "turnover",
            "portfolio_value", "allocation_entropy",
        ):
            mean, std = self._summary(self._values.get(key, []))
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        row.update(self._q_metrics())

        write_header = not self._header_written and not self.output_path.exists()
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._header_written = True
