from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch


class ContinuousTradingEnvironment:
    """Continuous-action trading environment with explicit cash/position accounting."""

    def __init__(
        self,
        data: pd.DataFrame,
        reward: str,
        window_size: int = 24,
        trading_period: Optional[int] = None,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 2.0,
        allow_short: bool = False,
        max_leverage: float = 1.0,
        action_low: float = 0.0,
        action_high: float = 1.0,
        min_equity_ratio: float = 0.2,
        stop_on_bankruptcy: bool = True,
        device: str = "auto",
    ) -> None:
        self.data = data
        valid_rewards = {"profit", "sr", "sr_enhanced"}
        if reward not in valid_rewards:
            raise ValueError(f"reward must be one of {sorted(valid_rewards)}")
        self.reward_f = reward
        self.window_size = int(window_size)
        self.trading_period = trading_period
        self.initial_capital = float(initial_capital)
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0.")
        self.transaction_cost_rate = float(transaction_cost_bps) / 10_000.0
        self.slippage_rate = float(slippage_bps) / 10_000.0
        if self.transaction_cost_rate < 0 or self.slippage_rate < 0:
            raise ValueError("transaction_cost_bps and slippage_bps must be >= 0.")
        self.allow_short = bool(allow_short)
        self.max_leverage = float(max_leverage)
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0.")
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        if self.action_high <= self.action_low:
            raise ValueError("action_high must be greater than action_low.")
        self.min_equity_ratio = float(min_equity_ratio)
        if not (0.0 < self.min_equity_ratio <= 1.0):
            raise ValueError("min_equity_ratio must be in (0, 1].")
        self.stop_on_bankruptcy = bool(stop_on_bankruptcy)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.rng = np.random.default_rng()
        self.last_start_index: Optional[int] = None
        self.last_start_timestamp: Optional[str] = None
        self._episode_end_index: Optional[int] = None
        self.sr_window = 30
        self.sr_eps = 1e-8
        self.sr_clip = 1.0
        if self.sr_window < 2:
            raise ValueError("sr_window must be >= 2 for sample std (ddof=1).")
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

    def _coerce_action(self, action: float | np.ndarray | torch.Tensor) -> float:
        if isinstance(action, torch.Tensor):
            values = action.detach().cpu().numpy().reshape(-1)
            return float(values[0]) if values.size > 0 else 0.0
        if isinstance(action, np.ndarray):
            values = action.reshape(-1)
            return float(values[0]) if values.size > 0 else 0.0
        return float(action)

    def _current_price(self) -> float:
        index = min(max(self.t, 0), len(self.data) - 1)
        return float(self.data.iloc[index]["Close"])

    def _position_value(self, price: float) -> float:
        return float(self.position_units * price)

    def _equity(self, price: float) -> float:
        return float(self.cash + self._position_value(price))

    def reset(
        self,
        seed: Optional[int] = None,
        start_index: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
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
        if self.trading_period is not None:
            self._episode_end_index = start_index + self.trading_period - 1
        else:
            self._episode_end_index = len(self.data) - 1

        self.cash = float(self.initial_capital)
        self.position_units = 0.0
        self.last_action = 0.0
        self.total_turnover = 0.0
        self.total_fees = 0.0
        self.prev_sr: Optional[float] = None
        self.ret_hist: list[float] = []
        self.cumulative_return = [0.0 for _ in range(len(self.data))]

        start_price = self._current_price()
        self.init_price = start_price
        self.equity_start = float(self.initial_capital)
        self.equity_end = float(self.initial_capital)
        self.prev_equity = float(self.initial_capital)
        self.realized_pnl = 0.0
        self.agent_open_position_value = self._position_value(start_price)

    def get_state(self) -> Optional[torch.Tensor]:
        if self.done:
            return None
        window = self.data.iloc[self.t - (self.window_size - 1) : self.t + 1]["Close"]
        return torch.tensor([float(el) for el in window], device=self.device, dtype=torch.float32)

    def get_account_features(self) -> dict[str, float]:
        price = self._current_price()
        equity = max(self._equity(price), self.sr_eps)
        position_value = self._position_value(price)
        return {
            "cash_ratio": float(self.cash / equity),
            "position_ratio": float(position_value / equity),
            "equity_return": float((equity / self.equity_start) - 1.0),
            "last_action": float(self.last_action),
            "turnover_ratio": float(self.total_turnover / max(self.equity_start, self.sr_eps)),
        }

    def _compute_reward(self, step_return: float) -> float:
        reward = 0.0
        if self.reward_f == "profit":
            reward = step_return
        elif self.reward_f == "sr":
            self.ret_hist.append(float(step_return))
            if len(self.ret_hist) >= self.sr_window:
                window = np.array(self.ret_hist[-self.sr_window :], dtype=float)
                mu = float(np.mean(window))
                sigma = float(np.std(window, ddof=1))
                reward = mu / (sigma + self.sr_eps)
            else:
                reward = 0.0
        elif self.reward_f == "sr_enhanced":
            self.ret_hist.append(float(step_return))
            if len(self.ret_hist) < self.sr_window:
                reward = 0.0
            else:
                window = np.array(self.ret_hist[-self.sr_window :], dtype=float)
                mu = float(np.mean(window))
                sigma = float(np.std(window, ddof=1))
                sr_t = mu / (sigma + self.sr_eps)
                reward = 0.0 if self.prev_sr is None else sr_t - self.prev_sr
                self.prev_sr = sr_t
            if self.sr_clip is not None and self.sr_clip > 0:
                reward = float(np.clip(reward, -self.sr_clip, self.sr_clip))
        return float(reward)

    def step(self, action: float | np.ndarray | torch.Tensor):
        if self.done:
            return (
                torch.tensor([0.0], device=self.device, dtype=torch.float32),
                True,
                torch.tensor([self._current_price()], device=self.device, dtype=torch.float32),
            )

        current_price = self._current_price()
        pre_equity = self._equity(current_price)
        if pre_equity <= self.sr_eps:
            self.done = True
            return (
                torch.tensor([-1.0], device=self.device, dtype=torch.float32),
                True,
                torch.tensor([current_price], device=self.device, dtype=torch.float32),
            )

        action_value = float(np.clip(self._coerce_action(action), self.action_low, self.action_high))
        if not self.allow_short:
            action_value = max(action_value, 0.0)
        action_value = float(np.clip(action_value, -self.max_leverage, self.max_leverage))

        target_notional = action_value * pre_equity
        target_units = target_notional / max(current_price, self.sr_eps)
        trade_units = target_units - self.position_units

        trade_notional = abs(trade_units) * current_price
        fee = trade_notional * self.transaction_cost_rate

        if trade_units > 0:
            buy_price = current_price * (1.0 + self.slippage_rate)
            required_cash = trade_units * buy_price + fee
            if required_cash > self.cash:
                unit_cost = buy_price + current_price * self.transaction_cost_rate
                feasible_units = max(self.cash / max(unit_cost, self.sr_eps), 0.0)
                trade_units = feasible_units
                target_units = self.position_units + trade_units
                trade_notional = abs(trade_units) * current_price
                fee = trade_notional * self.transaction_cost_rate
                required_cash = trade_units * buy_price + fee
            self.cash -= required_cash
        elif trade_units < 0:
            sell_units = -trade_units
            sell_price = current_price * (1.0 - self.slippage_rate)
            proceeds = sell_units * sell_price
            self.cash += proceeds - fee

        self.position_units = float(target_units)
        self.last_action = action_value
        self.total_turnover += float(trade_notional)
        self.total_fees += float(fee)

        post_equity = self._equity(current_price)
        self.agent_open_position_value = self._position_value(current_price)
        self.realized_pnl = post_equity - self.equity_start
        self.equity_end = post_equity

        step_return = (post_equity - self.prev_equity) / max(self.prev_equity, self.sr_eps)
        reward = self._compute_reward(step_return)
        self.prev_equity = post_equity
        self.cumulative_return[self.t] = (post_equity / max(self.equity_start, self.sr_eps)) - 1.0

        if self.stop_on_bankruptcy and post_equity <= self.equity_start * self.min_equity_ratio:
            self.done = True

        self.t += 1
        if self._episode_end_index is not None and self.t >= self._episode_end_index:
            self.done = True

        return (
            torch.tensor([reward], device=self.device, dtype=torch.float32),
            self.done,
            torch.tensor([current_price], device=self.device, dtype=torch.float32),
        )
