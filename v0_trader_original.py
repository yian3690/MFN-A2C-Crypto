import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from enum import Enum


class Trader:

    def __init__(self, starting_balance=10000, starting_price=[0, 0, 0, 0, 1], starting_action=[0, 0, 0, 0, 1]):
        np.set_printoptions(precision=4, suppress=True)
        self.starting_balance = starting_balance
        self.starting_price = starting_price
        self.asset = [0, 0, 0, 0, starting_balance]
        self.current_price = starting_price
        self.current_action = self.starting_action = starting_action
        self.portfolio_value = np.dot(self.asset, self.current_price)
        self.reset()

    def reset(self, seed=None):
        self.current_price = self.starting_price
        self.current_action = self.starting_action
        self.asset = [0, 0, 0, 0, self.starting_balance]
        self.portfolio_value = np.dot(self.asset, self.current_price)
        random.seed(seed)

    def perform_action(self, trader_action, crypto_price):
        self.current_price = crypto_price
        self.current_action = F.softmax(torch.Tensor(trader_action), dim=0).numpy()
        self.portfolio_value = np.dot(self.asset, self.current_price)

        self.portfolio_value = np.dot(self.asset, self.current_price)

    def output(self):
        print('Current Action: ' + str(self.current_action))
        print('Current balance: ' + str(self.portfolio_value))
        print('Including \nCash: ' + str(self.asset[4]) + '\nCrypto: ' + str(self.asset[0:4]) +
              '\n\nCurrent price: ' + str(self.current_price))
        print('------------------------------------------------------')


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print('This is a unit test')
    trader = Trader()
    trader.output()

    raw_data = pd.read_csv("merged_output.csv")
    episode_starting_step = 0
    actual_price_data = raw_data[[col for col in raw_data.columns if col.startswith('Open')]]
    temp = 0
    for i in range(10):
        action = np.random.rand(5)
        action = action / action.sum()
        price = actual_price_data.iloc[i]
        temp += trader.perform_action(action, np.append(price, 1))
        trader.output()
        print(temp)
