"""
SB3-compatible cryptocurrency portfolio environment.

This version keeps the user's original idea but fixes the major integration
points needed by Stable-Baselines3:

- Gymnasium reset()/step() signatures
- Correct two-view observation shape: 20 x (16 price + 16 indicators)
- 5-asset action: BTC, ETH, LTC, BNB, USDT
- Portfolio rebalancing at every 4-hour step
- Optional DSR reward
- No transaction fee, matching the paper's current environment
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class CryptoPortfolioEnv(gym.Env):
    """Gymnasium market simulator shared by the A2C and MFN-A2C experiments.

    Each step is one four-hour bar.  Five policy logits become non-negative
    weights for BTC, ETH, LTC, BNB and USDT; USDT is the risk-off asset with
    a zero return in this simplified paper-aligned environment.
    """
    metadata = {"render_modes": ["human"]}

    ASSETS = ["BTC", "ETH", "LTC", "BNB", "USDT"]

    def __init__(
        self,
        pct_csv: str,
        ta_csv: str,
        raw_csv: str,
        n_previous_timesteps: int = 20,
        max_episode_steps: int | None = None,
        reward_type: str = "dsr",
        initial_balance: float = 10000.0,
        eta: float = 0.005,
        random_start: bool = True,
        render_mode: str | None = None,
    ):
        """載入對齊資料，驗證特徵維度，並建立環境狀態與空間。"""
        super().__init__()

        self.pct_data = pd.read_csv(pct_csv).astype(np.float32)
        self.ta_data = pd.read_csv(ta_csv).astype(np.float32)
        self.raw_data = pd.read_csv(raw_csv)

        if len(self.pct_data) != len(self.ta_data) or len(self.pct_data) != len(self.raw_data):
            raise ValueError("pct/TA/raw CSV row counts do not match.")

        self.n_previous_timesteps = n_previous_timesteps
        self.initial_balance = float(initial_balance)
        self.eta = float(eta)
        self.reward_type = reward_type.lower()
        self.random_start = random_start
        self.render_mode = render_mode

        # The paper has 4 crypto assets; USDT is the fifth allocation.
        self.crypto_open_cols = [
            "Open0", "Open1", "Open2", "Open3"
        ]

        missing = [c for c in self.crypto_open_cols if c not in self.raw_data.columns]
        if missing:
            raise ValueError(f"Missing raw price columns: {missing}")

        self.price_data = self.raw_data[self.crypto_open_cols].to_numpy(
            dtype=np.float64
        )

        # 4 crypto x 4 features = 16; two modalities = 32.
        self.price_dim = self.pct_data.shape[1]
        self.indicator_dim = self.ta_data.shape[1]
        total_features = self.price_dim + self.indicator_dim

        if self.price_dim != 16:
            raise ValueError(
                f"Expected 16 price-change features (4 x 4), got {self.price_dim}."
            )
        if self.indicator_dim != 16:
            raise ValueError(
                f"Expected 16 technical-indicator features (4 x 4), got {self.indicator_dim}."
            )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_previous_timesteps, total_features),
            dtype=np.float32,
        )

        # A2C's standard continuous action distribution is used.
        # The environment converts the 5 outputs into portfolio weights
        # with softmax, matching the user's existing Trader implementation.
        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(5,),
            dtype=np.float32,
        )

        self.max_episode_steps = (
            max_episode_steps
            if max_episode_steps is not None
            else len(self.pct_data) - n_previous_timesteps - 1
        )

        self.start_idx = 0
        self.counter = 0
        self.balance = self.initial_balance
        self.weights = np.array([0, 0, 0, 0, 1.0], dtype=np.float64)

        self.return_history: list[float] = []
        self.balance_history: list[float] = []
        self.reward_history: list[float] = []

        self.A = 0.0
        self.B = 0.0

    def _get_obs(self, idx: int) -> np.ndarray:
        """Return the historical window while preserving MFN feature order."""
        end = idx + self.n_previous_timesteps
        price_window = self.pct_data.iloc[idx:end].to_numpy(dtype=np.float32)
        ta_window = self.ta_data.iloc[idx:end].to_numpy(dtype=np.float32)

        return np.concatenate([price_window, ta_window], axis=1)

    def _softmax(self, action: np.ndarray) -> np.ndarray:
        """Map unconstrained policy logits to valid portfolio weights."""
        z = np.asarray(action, dtype=np.float64)
        z = z - np.max(z)
        exp_z = np.exp(z)
        weights = exp_z / np.sum(exp_z)
        return weights

    def reset(self, *, seed: int | None = None, options=None):
        """重設資產、DSR 統計量和本回合的起始市場位置。"""
        super().reset(seed=seed)

        # Training can start at multiple valid points; evaluation sets
        # random_start=False so every model sees the same held-out period.
        max_start = len(self.pct_data) - self.n_previous_timesteps - self.max_episode_steps - 1

        if self.random_start and max_start > 0:
            self.start_idx = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start_idx = 0

        self.counter = 0
        self.balance = self.initial_balance
        self.weights = np.array([0, 0, 0, 0, 1.0], dtype=np.float64)

        self.return_history = []
        self.balance_history = [self.balance]
        self.reward_history = []

        self.A = 0.0
        self.B = 0.0

        obs = self._get_obs(self.start_idx)
        info = {"starting_idx": self.start_idx}

        return obs, info

    def _dsr_reward(self, portfolio_return: float) -> float:
        """Compute the paper's Differential Sharpe Ratio reward increment."""
        old_A = self.A
        old_B = self.B

        # A/B are exponentially weighted first/second moments of returns.
        # eta=0.005 in the paper controls how quickly DSR adapts.
        self.A = (1.0 - self.eta) * self.A + self.eta * portfolio_return
        self.B = (1.0 - self.eta) * self.B + self.eta * (portfolio_return ** 2)

        delta_A = self.A - old_A
        delta_B = self.B - old_B

        denominator = (old_B - old_A ** 2) ** 1.5

        if old_B <= old_A ** 2 + 1e-12 or denominator <= 1e-12:
            return 0.0

        return float(
            (old_B * delta_A - 0.5 * old_A * delta_B) / denominator
        )

    def step(self, action):
        """執行一次再平衡、計算下一期報酬與 reward，並前進一根 K 線。"""
        # 觀測視窗只包含截至 t-1 已完成的 K 線；在 Open[t] 再平衡，
        # 接著以 Open[t] 到 Open[t+1] 的價格變化計算本期報酬，避免偷看未來。
        decision_idx = (
            self.start_idx
            + self.counter
            + self.n_previous_timesteps
        )
        next_idx = decision_idx + 1

        if next_idx >= len(self.price_data):
            raise RuntimeError("Environment reached the end of the dataset.")

        # Rebalance for the next interval.  Fees and slippage are omitted,
        # as explicitly assumed by the current paper experiment.
        self.weights = self._softmax(action)

        # Four crypto returns from t -> t+1.
        current_prices = self.price_data[decision_idx]
        next_prices = self.price_data[next_idx]

        # Use raw Open prices to realise the next-period return.  The model
        # itself observes only transformed price and indicator inputs.
        crypto_returns = (next_prices / current_prices) - 1.0
        all_returns = np.concatenate([crypto_returns, [0.0]])

        portfolio_return = float(np.dot(self.weights, all_returns))
        self.balance *= (1.0 + portfolio_return)

        self.return_history.append(portfolio_return)
        self.balance_history.append(self.balance)

        dsr = self._dsr_reward(portfolio_return)

        if self.reward_type == "dsr":
            reward = dsr
        elif self.reward_type in {"pv", "value"}:
            reward = float(self.balance)
        elif self.reward_type in {"delta_pv", "delta"}:
            reward = float(self.balance_history[-1] - self.balance_history[-2])
        else:
            raise ValueError(
                "reward_type must be 'dsr', 'pv', or 'delta_pv'."
            )

        self.reward_history.append(reward)
        self.counter += 1

        terminated = self.counter >= self.max_episode_steps
        truncated = False

        next_obs_start = self.start_idx + self.counter
        if next_obs_start + self.n_previous_timesteps <= len(self.pct_data):
            obs = self._get_obs(next_obs_start)
        else:
            obs = self._get_obs(
                len(self.pct_data) - self.n_previous_timesteps
            )

        info = {
            "portfolio_value": float(self.balance),
            "portfolio_return": portfolio_return,
            "DSR": dsr,
            "weights": self.weights.copy(),
            "step": self.counter,
            "decision_idx": decision_idx,
            "next_idx": next_idx,
        }

        if terminated and self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """在終端機顯示目前步數、投資組合價值和配置權重。"""
        print(
            f"step={self.counter} | "
            f"PV={self.balance:.2f} | "
            f"weights={np.round(self.weights, 4)}"
        )

    def get_results(self) -> pd.DataFrame:
        """整理歷史投資組合價值、報酬與 reward，供評估腳本儲存。"""
        return pd.DataFrame({
            "portfolio_value": self.balance_history,
            "return": [np.nan] + self.return_history,
            "reward": [np.nan] + self.reward_history,
        })
