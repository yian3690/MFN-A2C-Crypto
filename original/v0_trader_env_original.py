import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import random

import v0_trader as trader
import numpy as np
import pandas as pd
import MFN

register(
    id='trader_v0',
    entry_point='v0_trader_env:TraderEnv'
)


def sharpe(ls):
    return np.mean(ls) / np.std(ls)


class TraderEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, n_previous_timesteps=20, n_indicators=4, n_timeseries=4, enable_indicator=True,
                 enable_MFN = False, max_episode_steps=180, reward_type=1, output=True, training=False):
        self.n_previous_timesteps = n_previous_timesteps
        self.n_indicators = n_indicators
        self.n_timeseries = n_timeseries
        self.trader = trader.Trader()
        self.enable_indicator = enable_indicator
        self.reward_type = reward_type  # 0 for ABS, 1 for DSR
        self.training = training
        self.trading_fee = 0
        self.enable_MFN = enable_MFN

        if training:
            self.pct_data = pd.read_csv("pct_change_output_train.csv")
            self.pct_data_w_ti = pd.concat([self.pct_data, pd.read_csv("ta_test_train.csv")], axis=1)
            self.raw_data = pd.read_csv("merged_output_train.csv")
            self.max_episode_steps = max_episode_steps
        else:
            self.pct_data = pd.read_csv("pct_change_output_test.csv")
            self.pct_data_w_ti = pd.concat([self.pct_data, pd.read_csv("ta_test_test.csv")], axis=1)
            self.raw_data = pd.read_csv("merged_output_test.csv")
            self.max_episode_steps = len(self.pct_data)-20

        self.actual_price_data = self.raw_data[[col for col in self.raw_data.columns if col.startswith('Open')]]
        self.episode_starting_step = None
        self.counter = 0
        self.episode_counter = 0

        self.output = output
        self.info = None
        self.record_balance = [self.trader.starting_balance]
        self.record_reward = []
        self.reward_now = 0
        self.episode_reward = []
        self.record_DSR = []
        self.record_cumDSR = []
        self.cumDSR = 0
        self.MFN = MFN

        self.action_space = spaces.Box(low=0, high=1, shape=(n_timeseries + 1,), dtype=np.float64)
        if self.enable_indicator:
            # self.observation_space = spaces.Dict({
            #     "price_pct": spaces.Box(low=-50, high=50, shape=(n_previous_timesteps, 4 * self.n_timeseries),
            #                             dtype=np.float64),
            #     "indicator": spaces.Box(low=-50, high=50, shape=(n_indicators,), dtype=np.float64),
            # })
            self.observation_space = spaces.Box(low=-50, high=50, shape=(n_previous_timesteps, 8 * self.n_timeseries),
                                                dtype=np.float64)
        elif self.enable_MFN:
            self.observation_space = spaces.Box(low=-50, high=50, shape=(n_previous_timesteps, 2 * 8 * self.n_timeseries),
                                                dtype=np.float64)
        else:
            self.observation_space = spaces.Box(low=-50, high=50, shape=(n_previous_timesteps, 4 * self.n_timeseries),
                                                dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.trader.reset(seed=seed)
        self.counter = 0
        self.record_DSR = []
        self.record_balance = [self.trader.starting_balance]
        self.record_reward = []
        self.reward_now = 0
        self.record_cumDSR = []
        self.cumDSR = 0
        self.trading_fee = 0

        max_start_idx = self.pct_data.shape[0] - self.n_previous_timesteps - self.max_episode_steps - 1

        #self.episode_starting_step = 0
        if self.training:
            self.episode_starting_step = pd.Series(range(max_start_idx + 1)).sample(n=1, random_state=seed).iloc[0]
        else:
            self.episode_starting_step = 0

        if self.enable_indicator:
            # TODO
            # obs = {"price_pct": self.pct_data[
            #                     self.episode_starting_step:self.episode_starting_step + self.n_previous_timesteps],
            #        "indicator": np.array(self.n_indicators)
            #        }
            obs = self.pct_data_w_ti.iloc[self.episode_starting_step:self.episode_starting_step + self.n_previous_timesteps]
            obs = obs.to_numpy()

        else:
            obs = self.pct_data.iloc[self.episode_starting_step:self.episode_starting_step + self.n_previous_timesteps]
            obs = obs.to_numpy()

        info = {"Starting idx": self.episode_starting_step}
        return obs, info

    def step(self, action):
        current_price = self.actual_price_data.iloc[
            self.episode_starting_step + self.n_previous_timesteps + self.counter - 1]
        observation_price = self.actual_price_data.iloc[
            self.episode_starting_step + self.n_previous_timesteps + self.counter]

        self.trading_fee += self.trader.perform_action(action, np.append(current_price, 1))

        terminated = False
        current_balance = np.dot(self.trader.asset, np.append(observation_price, 1))

        current_DSR = self.cal_DSR(self.record_balance, current_balance)
        self.record_DSR.append(current_DSR)
        self.cumDSR += current_DSR
        self.record_cumDSR.append(self.cumDSR)

        if self.reward_type == 0:
            reward = current_balance
        elif self.reward_type == 1:
            # TODO
            reward = self.cumDSR
        elif self.reward_type == 2:
            reward = current_balance - self.record_balance[-1]
        else:
            # TODO
            reward = current_balance

        self.record_balance = np.append(self.record_balance, current_balance)

        self.counter = self.counter + 1
        self.record_reward.append(reward)
        # obs = self.pct_data.iloc[self.episode_starting_step + self.counter:
        #                          self.episode_starting_step + self.counter + self.n_previous_timesteps]
        # obs = obs.to_numpy()
        if self.enable_indicator:
            # TODO
            # obs = {"price_pct": self.pct_data[
            #                     self.episode_starting_step:self.episode_starting_step + self.n_previous_timesteps],
            #        "indicator": np.array(self.n_indicators)
            #        }
            obs = pct_data_w_ti.iloc[self.episode_starting_step + self.counter:
                                     self.episode_starting_step + self.counter + self.n_previous_timesteps]
            obs = obs.to_numpy()
        elif self.enable_MFN:
            self.MFN.train_mfn(self.pct_data_w_ti.iloc[self.episode_starting_step + self.counter:
                                     self.episode_starting_step + self.counter + self.n_previous_timesteps])

        else:
            obs = self.pct_data.iloc[self.episode_starting_step + self.counter:
                                     self.episode_starting_step + self.counter + self.n_previous_timesteps]
            obs = obs.to_numpy()

        if self.counter >= self.max_episode_steps:
            terminated = True
            df = pd.DataFrame(self.record_balance)
            df.to_csv("portfolio_value.csv", index=False)
            df = pd.DataFrame(self.record_reward)
            df.to_csv("reward.csv", index=False)
            df = pd.DataFrame(self.record_DSR)
            df.to_csv("DSR.csv", index=False)
            df = pd.DataFrame(self.record_cumDSR)
            df.to_csv("cumDSR.csv", index=False)
            # self.temp_reward = self.temp_reward - self.trader.starting_balance
            # self.episode_reward = np.append(self.episode_reward, self.temp_reward.sum())
            # print(self.episode_reward[self.episode_counter])
            self.episode_counter = self.episode_counter + 1

        basic = {"counter": self.counter,
                 "episode counter": self.episode_counter,
                 "step": self.episode_starting_step + self.n_previous_timesteps + self.counter - 1,
                 "reward": reward,
                 "portfolio_value": current_balance,
                 "DSR": self.cumDSR
                 }
        extra = {
            "episode_reward": self.episode_reward
        }
        info = {"basic": basic,
                "extra": extra,
                "fee": self.trading_fee
                }
        self.info = info

        if self.output and self.counter % 100 == 0:
            self.render()

        if self.episode_counter == 100 and False:
            # df = pd.DataFrame(self.episode_reward)
            # df.transpose()
            df = pd.read_csv('reward.csv', header=None)
            df.loc[len(df)] = self.episode_reward
            df.to_csv('reward.csv', index=False, header=False)


        return obs, reward, terminated, False, info

    def render(self):
        # self.trader.output()
        print(self.info['basic'])

    def get_reward(self):
        return self.episode_reward

    def cal_DSR(self, rec, current_value):
        if self.counter < 5:
            return 0
        else:
            eta = 0.005
            current_pct = (current_value/rec[-1])-1
            pct = pd.DataFrame(rec).pct_change()
            A = np.mean(pct)
            B = np.mean(pct ** 2)
            delta_A = current_pct - A
            delta_B = current_pct ** 2 - B
            Dt = (B * delta_A - 0.5 * A * delta_B) / (B - A ** 2) ** (3 / 2)
        return eta * Dt

    def MFN_config(self):
        config = dict()
        config["input_dims"] = [5, 20]
        # hl = random.choice([32, 64, 88, 128, 156, 256])
        ha = random.choice([8, 16, 32, 48, 64, 80])
        hv = random.choice([8, 16, 32, 48, 64, 80])
        config["h_dims"] = [ha, hv]
        config["memsize"] = random.choice([64, 128, 256, 300, 400])
        config["windowsize"] = 2
        config["batchsize"] = random.choice([32, 64, 128, 256])
        config["num_epochs"] = 50
        config["lr"] = random.choice([0.001, 0.002, 0.005, 0.008, 0.01])
        config["momentum"] = random.choice([0.1, 0.3, 0.5, 0.6, 0.8, 0.9])
        NN1Config = dict()
        NN1Config["shapes"] = random.choice([32, 64, 128, 256])
        NN1Config["drop"] = random.choice([0.0, 0.2, 0.5, 0.7])
        NN2Config = dict()
        NN2Config["shapes"] = random.choice([32, 64, 128, 256])
        NN2Config["drop"] = random.choice([0.0, 0.2, 0.5, 0.7])
        gamma1Config = dict()
        gamma1Config["shapes"] = random.choice([32, 64, 128, 256])
        gamma1Config["drop"] = random.choice([0.0, 0.2, 0.5, 0.7])
        gamma2Config = dict()
        gamma2Config["shapes"] = random.choice([32, 64, 128, 256])
        gamma2Config["drop"] = random.choice([0.0, 0.2, 0.5, 0.7])
        outConfig = dict()
        outConfig["shapes"] = random.choice([32, 64, 128, 256])
        outConfig["drop"] = random.choice([0.0, 0.2, 0.5, 0.7])
        configs = [config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig]
        return configs


if __name__ == "__main__":
    env = gym.make('trader_v0')
    _, info = env.reset()
    np.set_printoptions(precision=4, suppress=True)
    print(info)
    print('This is a unit test')

    raw_data = pd.read_csv("merged_output.csv")
    episode_starting_step = 0
    actual_price_data = raw_data[[col for col in raw_data.columns if col.startswith('Open')]]

    for i in range(500):
        action = np.random.rand(5)
        action = action / action.sum()
        obs, reward, _, _, info = env.step(action)
        # print(info)
