from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch


class TradingEnvironment:
    """Trading environment for the D3QN agent."""

    def __init__(
        self,
        data: pd.DataFrame,
        reward: str,
        window_size: int = 24,
        device: str = "auto",
    ) -> None:
        self.data = data
        self.reward_f = reward if reward == "sr" else "profit"
        self.window_size = window_size
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.reset()

    def reset(self) -> None:
        """Reset the environment. Must be called before step()."""
        self.t = self.window_size - 1
        self.done = False
        self.profits = [0 for _ in range(len(self.data))]
        self.agent_positions = []
        self.agent_open_position_value = 0
        self.cumulative_return = [0 for _ in range(len(self.data))]
        self.init_price = self.data.iloc[0, :]["Close"]

    def get_state(self) -> Optional[torch.Tensor]:
        """Return the current state of the environment."""
        if self.done:
            return None
        window = self.data.iloc[self.t - (self.window_size - 1) : self.t + 1, :]["Close"]
        return torch.tensor([el for el in window], device=self.device, dtype=torch.float)

    def step(self, act: int | torch.Tensor):
        """Perform the action and return (reward, done, state)."""
        if isinstance(act, torch.Tensor):
            act = int(act.item())

        reward = 0
        state = self.data.iloc[self.t, :]["Close"]

        if act == 0:  # Do nothing
            pass

        if act == 1:  # Buy
            self.agent_positions.append(self.data.iloc[self.t, :]["Close"])

        sell_nothing = False
        if act == 2:  # Sell
            profits = 0
            if len(self.agent_positions) < 1:
                sell_nothing = True
            for position in self.agent_positions:
                profits += self.data.iloc[self.t, :]["Close"] - position

            self.profits[self.t] = profits
            self.agent_positions = []

        self.agent_open_position_value = 0
        for position in self.agent_positions:
            self.agent_open_position_value += self.data.iloc[self.t, :]["Close"] - position
            self.cumulative_return[self.t] += (position - self.init_price) / self.init_price

        reward = 0
        if self.reward_f == "sr":
            std_close = np.std(np.array(self.data.iloc[0 : self.t]["Close"]))
            sr = (self.agent_open_position_value + 0.024) / std_close if std_close != 0 else 0
            if sr <= -4:
                reward = -20
            elif sr < -1:
                reward = -10
            elif sr < 0:
                reward = -5
            elif sr == 0:
                reward = 0
            elif sr <= 1:
                reward = 5
            elif sr < 4:
                reward = 10
            else:
                reward = 20

        if self.reward_f == "profit":
            profit_value = self.profits[self.t]
            if profit_value > 0:
                reward = 5
            elif profit_value < 0:
                reward = -5
            elif profit_value == 0:
                reward = 0

        if sell_nothing and (reward > -5):
            reward = -5
        if act == 0:
            reward = 0.5
        if reward < -100:
            self.done = True

        self.t += 1
        if self.t == len(self.data) - 1:
            self.done = True

        return (
            torch.tensor([reward], device=self.device, dtype=torch.float),
            self.done,
            torch.tensor([state], device=self.device, dtype=torch.float),
        )
