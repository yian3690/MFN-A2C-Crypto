"""
SB3-compatible cryptocurrency portfolio environment.

This version keeps the user's original idea but fixes the major integration
points needed by Stable-Baselines3:

- Gymnasium reset()/step() signatures
- Paper-style two-view observation shape: 20 x (5 price + 20 indicators)
- 5-asset action: BTC, ETH, LTC, BNB, USDT
- Portfolio rebalancing at every 2-hour step
- Optional DSR reward
- No transaction fee, matching the paper's current environment
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.dsr import DEFAULT_FORMULA, DEFAULT_WARMUP_STEPS, DSRTracker
from src.feature_schema import INDICATOR_DIM, PORTFOLIO_ASSETS, PRICE_DIM


class CryptoPortfolioEnv(gym.Env):
    """Gymnasium market simulator shared by the A2C and MFN-A2C experiments.

    Each step is one two-hour bar. In simplex mode the action is already a
    non-negative, unit-sum weight vector. Legacy logits mode remains
    available for experiments that have not yet migrated.
    """
    metadata = {"render_modes": ["human"]}

    ASSETS = list(PORTFOLIO_ASSETS)

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
        dsr_warmup_steps: int = DEFAULT_WARMUP_STEPS,
        dsr_formula: str = DEFAULT_FORMULA,
        random_start: bool = True,
        action_mode: str = "logits",
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
        self.dsr_warmup_steps = int(dsr_warmup_steps)
        self.dsr_formula = dsr_formula
        self.reward_type = reward_type.lower()
        self.random_start = random_start
        self.action_mode = action_mode.lower()
        self.render_mode = render_mode

        if self.action_mode not in {"logits", "simplex"}:
            raise ValueError("action_mode must be 'logits' or 'simplex'.")

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

        # Thesis schema: five price relatives and four indicators per asset.
        self.price_dim = self.pct_data.shape[1]
        self.indicator_dim = self.ta_data.shape[1]
        total_features = self.price_dim + self.indicator_dim

        if self.price_dim != PRICE_DIM:
            raise ValueError(
                f"Expected {PRICE_DIM} price-relative features, got {self.price_dim}."
            )
        if self.indicator_dim != INDICATOR_DIM:
            raise ValueError(
                f"Expected {INDICATOR_DIM} technical-indicator features "
                f"(5 x 4), got {self.indicator_dim}."
            )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_previous_timesteps, total_features),
            dtype=np.float32,
        )

        if self.action_mode == "simplex":
            # SimplexActorCriticPolicy samples these weights directly from a
            # Dirichlet distribution. Box describes per-component bounds;
            # _validate_simplex_action additionally enforces sum(weights)=1.
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5,),
                dtype=np.float32,
            )
        else:
            # Backward-compatible mode for A2C experiments that still emit
            # unconstrained logits and transform them with softmax.
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
        self.weight_history: list[np.ndarray] = []
        self.turnover_history: list[float] = []

        self.dsr_tracker = DSRTracker(
            eta=self.eta,
            warmup_steps=self.dsr_warmup_steps,
            formula=self.dsr_formula,
        )

    def _get_obs(self, idx: int) -> np.ndarray:
        """Return the historical window while preserving MFN feature order."""
        end = idx + self.n_previous_timesteps
        price_window = self.pct_data.iloc[idx:end].to_numpy(dtype=np.float32)
        ta_window = self.ta_data.iloc[idx:end].to_numpy(dtype=np.float32)

        return np.concatenate([price_window, ta_window], axis=1)

    def _validate_simplex_action(self, action: np.ndarray) -> np.ndarray:
        """Validate and normalize only floating-point round-off on weights."""
        weights = np.asarray(action, dtype=np.float64).reshape(-1)
        if weights.shape != (len(self.ASSETS),):
            raise ValueError(
                f"Expected {len(self.ASSETS)} portfolio weights, "
                f"got shape {weights.shape}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("Portfolio weights must all be finite.")
        if np.any(weights < -1e-7) or np.any(weights > 1.0 + 1e-7):
            raise ValueError("Portfolio weights must be between 0 and 1.")

        total = float(weights.sum())
        if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError(
                "Portfolio weights must sum to 1; "
                f"received {total:.9f}."
            )

        weights = np.clip(weights, 0.0, 1.0)
        return weights / weights.sum()

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """Convert the configured policy action into portfolio weights."""
        if self.action_mode == "simplex":
            return self._validate_simplex_action(action)

        logits = np.asarray(action, dtype=np.float64).reshape(-1)
        if logits.shape != (len(self.ASSETS),):
            raise ValueError(
                f"Expected {len(self.ASSETS)} action logits, "
                f"got shape {logits.shape}."
            )
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

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
        self.weight_history = [self.weights.copy()]
        self.turnover_history = []

        self.dsr_tracker.reset()

        obs = self._get_obs(self.start_idx)
        info = {"starting_idx": self.start_idx}

        return obs, info

    def _dsr_reward(self, portfolio_return: float) -> float:
        """Return one increment from the project's shared DSR tracker."""
        return self.dsr_tracker.update(portfolio_return)

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
        previous_weights = self.weights.copy()
        self.weights = self._action_to_weights(action)
        # Half-L1 turnover counts reallocated capital only once.
        turnover = float(0.5 * np.abs(self.weights - previous_weights).sum())

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
        self.weight_history.append(self.weights.copy())
        self.turnover_history.append(turnover)

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
            "dsr_first_moment": self.dsr_tracker.first_moment,
            "dsr_second_moment": self.dsr_tracker.second_moment,
            "dsr_variance": (
                self.dsr_tracker.second_moment
                - self.dsr_tracker.first_moment**2
            ),
            "weights": self.weights.copy(),
            "turnover": turnover,
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
        """整理績效、配置權重與換手率，供評估腳本儲存。"""
        result = pd.DataFrame({
            "portfolio_value": self.balance_history,
            "return": [np.nan] + self.return_history,
            "reward": [np.nan] + self.reward_history,
            "turnover": [np.nan] + self.turnover_history,
        })
        weights = np.asarray(self.weight_history, dtype=np.float64)
        for index, asset in enumerate(self.ASSETS):
            result[f"weight_{asset.lower()}"] = weights[:, index]
        return result
