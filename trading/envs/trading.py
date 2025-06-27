import gymnasium as gym
import numpy as np
from gymnasium import spaces
import ccxt
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt


class TradingEnv(gym.Env):
    def __init__(self):
        self.ohlcv_df = self._get_data()

        self._current_step = 0

        self._current_price = 0
        self._crypto_balance = 0
        self._initial_fiat_balance = 1000  # USD
        self._fiat_balance = self._initial_fiat_balance

        self._reward = 0
        self._rewards = [self._reward]

        self._any_numbers = [0]

        # 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Dict({
            "price": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            "crypto_balance": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            "fiat_balance": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
        })

        # rendering
        self.fig = None
        self.ax = None
        self.line_distance = None
        self.line_reward = None
        # The size of Matplot window
        self.figsize = (10, 5)

    def _get_data(self):
        binance = ccxt.binance()
        one_year_ago = datetime.now() - timedelta(days=365)
        ohlcv = binance.fetch_ohlcv('BTC/USDT', timeframe='1m', since=int(one_year_ago.timestamp()))

        ohlcv_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        return ohlcv_df

    def _get_obs(self):
        return {
            "price": self._current_price,
            "crypto_balance": self._crypto_balance,
            "fiat_balance": self._fiat_balance,
        }

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        self._current_price = 0
        self._crypto_balance = 0
        self._current_step = 0

        self._fiat_balance = self._initial_fiat_balance

        return self._get_obs(), self._get_info()

    def step(self, action):
        self._current_price = self.ohlcv_df.loc[self._current_step, 'close']

        # buy
        if action == 1:
            if self._fiat_balance > 0:
                self._crypto_balance += (self._fiat_balance / self._current_price)
                self._fiat_balance = 0
        # sell
        elif action == 2:
            if self._crypto_balance > 0:
                self._fiat_balance += (self._crypto_balance * self._current_price)
                self._crypto_balance = 0

        self._current_step += 1
        next_price = self.ohlcv_df.loc[self._current_step, 'close']

        total_asset = self._fiat_balance + (self._crypto_balance * next_price)
        self._reward = total_asset - self._initial_fiat_balance
        self._rewards.append(self._reward)

        # self.render()

        done = self._current_step >= len(self.ohlcv_df) - 1

        return self._get_obs(), self._reward, done, False, self._get_info()

    def render(self):
        if self.fig is None:
            plt.ion()

            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self.line_reward, = self.ax.plot([], [], label="Reward", color='green', marker='o')
            self.ax.legend()

        self.line_reward.set_data(list(range(0, len(self._rewards))), self._rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)

    def render_any(self, any_number):
        self._any_numbers.append(any_number)

        if self.fig is None:
            plt.ion()

            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self.line_reward, = self.ax.plot([], [], label="Reward", color='green', marker='o')
            self.ax.legend()

        self.line_reward.set_data(list(range(0, len(self._any_numbers))), self._any_numbers)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)

    def batch_render_any(self, any_numbers, block: False):
        plt.plot(any_numbers)
        plt.show(block=block)
