"""DQN action adapter: converts a discrete asset choice into portfolio logits."""

import numpy as np
import gymnasium as gym


class DiscretePortfolioWrapper(gym.ActionWrapper):
    """
    Convert discrete DQN action into a 5-asset portfolio weight vector.

    0 -> BTC
    1 -> ETH
    2 -> LTC
    3 -> BNB
    4 -> USDT
    """

    def __init__(self, env):
        """設定 DQN 的五個離散動作與被包裝的交易環境。"""
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(5)

    def action(self, action):
        """將選定資產轉為環境 softmax 可辨識的近乎 one-hot logits。"""
        weights = np.zeros(5, dtype=np.float32)

        weights[int(action)] = 1.0

        # Existing CryptoPortfolioEnv receives action logits
        # and applies softmax internally.
        #
        # Therefore use a strong logit for selected asset.
        logits = np.full(5, -10.0, dtype=np.float32)
        logits[int(action)] = 10.0

        return logits