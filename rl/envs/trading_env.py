from __future__ import annotations

from typing import Optional, Tuple

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
        trading_period: Optional[int] = None,
        max_positions: Optional[int] = None,
        sell_mode: str = "all",
        device: str = "auto",
    ) -> None:
        self.data = data
        self.reward_f = reward if reward == "sr" else "profit"
        self.window_size = window_size
        self.trading_period = trading_period
        if max_positions is not None and max_positions < 1:
            raise ValueError("max_positions must be >= 1 or None.")
        valid_sell_modes = {"all", "one", "all_cap", "one_plus"}
        if sell_mode not in valid_sell_modes:
            raise ValueError("sell_mode must be 'all', 'one', 'all_cap', or 'one_plus'.")
        if sell_mode == "all_cap" and max_positions is None:
            raise ValueError("sell_mode='all_cap' requires max_positions to be set.")
        self.max_positions = int(max_positions) if max_positions is not None else None
        self.sell_mode = sell_mode
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.rng = np.random.default_rng()
        self.last_start_index: Optional[int] = None
        self.last_start_timestamp: Optional[str] = None
        self._episode_end_index: Optional[int] = None
        self.reset(start_index=self.window_size - 1)

    def _get_valid_start_range(self) -> Tuple[int, int]:
        min_start = self.window_size - 1
        max_start = len(self.data) - 1
        if self.trading_period is not None:
            max_start = len(self.data) - self.trading_period
        return min_start, max_start

    def _resolve_start_timestamp(self, start_index: int) -> Optional[str]:
        if "Date" not in self.data.columns:
            return None
        timestamp = self.data.iloc[start_index]["Date"]
        if isinstance(timestamp, pd.Timestamp):
            return timestamp.isoformat()
        return str(timestamp)

    def reset(
        self,
        seed: Optional[int] = None,
        start_index: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
        """Reset the environment. Must be called before step()."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        min_start, max_start = self._get_valid_start_range()
        if max_start < min_start:
            raise ValueError(
                "Invalid start_index range "
                f"[{min_start}, {max_start}] for data length {len(self.data)} "
                f"and trading_period {self.trading_period}."
            )

        if start_index is None:
            start_index = int(self.rng.integers(min_start, max_start + 1))
        else:
            if not (min_start <= start_index <= max_start):
                raise ValueError(
                    "start_index must be in the range "
                    f"[{min_start}, {max_start}] but got {start_index}."
                )
            start_index = int(start_index)

        self.last_start_index = start_index
        self.last_start_timestamp = self._resolve_start_timestamp(start_index)
        self.t = start_index
        self.done = False
        self.profits = [0 for _ in range(len(self.data))]
        self.agent_positions = []
        self.agent_open_position_value = 0
        self.cumulative_return = [0 for _ in range(len(self.data))]
        if self.trading_period is not None:
            self._episode_end_index = start_index + self.trading_period - 1
        else:
            self._episode_end_index = len(self.data) - 1
        self.init_price = self.data.iloc[start_index - (self.window_size - 1), :]["Close"]

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
        current_price = self.data.iloc[self.t, :]["Close"]
        state = current_price

        if act == 0:  # Do nothing
            pass

        if act == 1:  # Buy
            if self.max_positions is None or len(self.agent_positions) < self.max_positions:
                self.agent_positions.append(current_price)

        sell_nothing = False
        if act in {2, 3}:  # Sell (one or all depending on mode/action)
            if len(self.agent_positions) < 1:
                sell_nothing = True
            else:
                profits = 0.0
                sell_all = False
                if act == 2:
                    sell_all = self.sell_mode in {"all", "all_cap"}
                elif act == 3:
                    if self.sell_mode == "one_plus":
                        sell_all = True
                    else:
                        sell_all = self.sell_mode in {"all", "all_cap"}
                if sell_all:
                    for position in self.agent_positions:
                        profits += current_price - position
                    self.agent_positions = []
                else:
                    position = self.agent_positions.pop(0)
                    profits = current_price - position
                self.profits[self.t] = profits

        self.agent_open_position_value = 0
        for position in self.agent_positions:
            self.agent_open_position_value += current_price - position
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
        if self._episode_end_index is not None and self.t >= self._episode_end_index:
            self.done = True

        return (
            torch.tensor([reward], device=self.device, dtype=torch.float),
            self.done,
            torch.tensor([state], device=self.device, dtype=torch.float),
        )
