"""使用獨立Validation期間選擇最佳模型，絕不讀取Test資料。"""

from __future__ import annotations

import csv
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


def read_best_validation(log_path: str | Path) -> tuple[float, int | None]:
    """從Validation CSV讀取最高Final PV及其步數。"""
    path = Path(log_path)
    best_pv = float("-inf")
    best_timestep = None
    if not path.exists():
        return best_pv, best_timestep
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            try:
                candidate_pv = float(row["final_pv"])
                candidate_timestep = int(row["timesteps"])
            except (KeyError, TypeError, ValueError):
                continue
            if candidate_pv > best_pv:
                best_pv = candidate_pv
                best_timestep = candidate_timestep
    return best_pv, best_timestep


class ValidationBestModelCallback(BaseCallback):
    """定期完整回測Validation，依Final PV保存最佳checkpoint。

    Validation環境採固定時間順序且每次重設，因此不同checkpoint看到的
    市場期間完全相同。Final PV是唯一選模分數；episode reward只記錄，
    不作為Test調參或模型選擇依據。
    """

    def __init__(
        self,
        eval_env,
        *,
        eval_freq: int,
        best_model_path: str | Path,
        log_path: str | Path,
        reset_log: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        if eval_freq <= 0:
            raise ValueError("eval_freq必須大於0。")
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.best_model_path = Path(best_model_path)
        self.log_path = Path(log_path)
        self.reset_log = bool(reset_log)
        self.best_final_pv = float("-inf")
        self.best_timestep: int | None = None

    def _on_training_start(self) -> None:
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.reset_log and self.log_path.exists():
            self.log_path.unlink()
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "timesteps",
                        "final_pv",
                        "episode_reward",
                        "mean_step_reward",
                        "is_best",
                    ]
                )
        else:
            # --resume時沿用先前最佳分數，避免較差checkpoint覆蓋最佳模型。
            self.best_final_pv, self.best_timestep = read_best_validation(
                self.log_path
            )

    def _evaluate_once(self) -> tuple[float, float, float]:
        obs, _ = self.eval_env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        final_pv = float("nan")

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            episode_reward += float(reward)
            steps += 1
            final_pv = float(info["portfolio_value"])
            done = bool(terminated or truncated)

        mean_step_reward = episode_reward / max(steps, 1)
        return final_pv, episode_reward, mean_step_reward

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        final_pv, episode_reward, mean_step_reward = self._evaluate_once()
        is_best = final_pv > self.best_final_pv
        if is_best:
            self.best_final_pv = final_pv
            self.best_timestep = int(self.num_timesteps)
            self.model.save(str(self.best_model_path))

        with self.log_path.open("a", newline="", encoding="utf-8-sig") as file:
            csv.writer(file).writerow(
                [
                    self.num_timesteps,
                    f"{final_pv:.10f}",
                    f"{episode_reward:.10f}",
                    f"{mean_step_reward:.10f}",
                    int(is_best),
                ]
            )

        if self.verbose:
            status = "BEST SAVED" if is_best else "not improved"
            print(
                f"Validation @ {self.num_timesteps:,}: "
                f"Final PV={final_pv:.2f}, "
                f"episode reward={episode_reward:.6f} ({status})"
            )
        return True

    def close(self) -> None:
        self.eval_env.close()
