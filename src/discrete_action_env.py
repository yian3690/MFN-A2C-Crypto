"""DQN action adapter: converts a discrete choice into simplex weights."""

import itertools

import gymnasium as gym
import numpy as np


class DiscretePortfolioWrapper(gym.ActionWrapper):
    """
    Convert a discrete DQN action into a portfolio allocation.

    Default discretization:
        weight_step = 0.2

    For 5 assets, the unconstrained grid contains 126 allocations. With
    the default 60% cap on BTC/ETH/LTC/BNB, 106 allocations remain. USDT
    is uncapped so the policy may choose a fully risk-off allocation.
    """

    def __init__(
        self,
        env,
        weight_step=0.1,
        max_crypto_weight=0.4,
    ):
        super().__init__(env)

        self.weight_step = weight_step
        self.max_crypto_weight = float(max_crypto_weight)

        self.n_assets = 5
        self.crypto_asset_count = 4

        # 0.2 -> 5 units
        self.units = int(
            round(
                1.0 / self.weight_step
            )
        )

        self.portfolios = (
            self._generate_portfolios()
        )

        self.action_space = (
            gym.spaces.Discrete(
                len(self.portfolios)
            )
        )

        print(
            "DQN discrete portfolio actions:",
            len(self.portfolios),
        )
        print(
            "DQN maximum crypto weight:",
            f"{self.max_crypto_weight:.0%}",
        )

    def _generate_portfolios(self):

        portfolios = []

        for allocation in itertools.product(
            range(self.units + 1),
            repeat=self.n_assets,
        ):

            if sum(allocation) != self.units:
                continue

            weights = np.array(
                allocation,
                dtype=np.float32,
            ) / self.units

            # BTC/ETH/LTC/BNB are capped to reduce concentrated risk.
            # USDT (the final asset) may still reach 100% for risk-off use.
            if np.any(
                weights[:self.crypto_asset_count]
                > self.max_crypto_weight + 1e-8
            ):
                continue

            portfolios.append(
                weights
            )

        return np.array(
            portfolios,
            dtype=np.float32,
        )

    def action(self, action):

        action = int(action)

        if (
            action < 0
            or action >= len(self.portfolios)
        ):
            raise ValueError(
                f"Invalid DQN action: {action}"
            )

        weights = (
            self.portfolios[action]
        )

        return weights.astype(
            np.float32
        )
