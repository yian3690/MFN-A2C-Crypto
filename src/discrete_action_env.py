"""DQN action adapter: converts a discrete asset choice into portfolio logits."""

import itertools

import gymnasium as gym
import numpy as np


class DiscretePortfolioWrapper(gym.ActionWrapper):
    """
    Convert a discrete DQN action into a portfolio allocation.

    Default discretization:
        weight_step = 0.2

    For 5 assets, this generates 126 valid allocations
    satisfying:

        w_i >= 0
        sum(w_i) = 1
    """

    def __init__(
        self,
        env,
        weight_step=0.2,
    ):
        super().__init__(env)

        self.weight_step = weight_step

        self.n_assets = 5

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
            )

            weights /= self.units

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

        # CryptoPortfolioEnv applies softmax internally.
        # softmax(log(w)) ~= w.
        # epsilon prevents log(0).

        eps = 1e-8

        logits = np.log(
            np.clip(
                weights,
                eps,
                1.0,
            )
        )

        return logits.astype(
            np.float32
        )