"""Observation wrapper used to remove technical indicators for Experiment 1."""

import numpy as np
import gymnasium as gym


class PriceOnlyWrapper(gym.ObservationWrapper):
    """
    A2C without Technical Indicators.

    Original observation:
        [price-change features | technical-indicator features]

    New observation:
        [price-change features]

    Paper Experiment 1:
        A2C w/o TI uses only price-change information.
        MFN is not used because there is only one modality.
    """

    def __init__(self, env, price_dim=16):

        """依價格特徵數量重新宣告只有單一模態的觀察空間。"""
        super().__init__(env)

        self.price_dim = price_dim

        old_space = env.observation_space

        if len(old_space.shape) != 2:
            raise ValueError(
                f"Expected 2D observation space, "
                f"got {old_space.shape}"
            )

        timesteps = old_space.shape[0]

        if old_space.shape[1] < price_dim:
            raise ValueError(
                f"Observation has only "
                f"{old_space.shape[1]} features, "
                f"but price_dim={price_dim}."
            )

        # Keep only the first 16 price-change features.
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(timesteps, price_dim),
            dtype=np.float32,
        )

    def observation(self, observation):

        """切出每個時間點的價格變化特徵，丟棄技術指標。"""
        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        # Assumption:
        # first 16 = price-change modality
        # last 16 = technical-indicator modality
        return observation[:, :self.price_dim]