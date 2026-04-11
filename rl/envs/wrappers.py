"""Environment wrappers for trading environments."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from rl.features.signature import LogSigTransformer, PathBuilder


class SignatureObsWrapper:
    """Replaces raw price windows with logsignature observations."""

    def __init__(
        self,
        env,
        path_builder: PathBuilder,
        logsig_transformer: LogSigTransformer,
        add_position: bool = False,
        account_feature_keys: Optional[list[str]] = None,
        subwindow_sizes: Optional[list[int]] = None,
    ) -> None:
        self.env = env
        self.path_builder = path_builder
        self.logsig_transformer = logsig_transformer
        self.add_position = add_position
        self.account_feature_keys = list(account_feature_keys or [])
        self.subwindow_sizes = [int(size) for size in (subwindow_sizes or [])]
        if any(size < 2 for size in self.subwindow_sizes):
            raise ValueError("signature.subwindow_sizes must contain values >= 2.")
        if len(set(self.subwindow_sizes)) != len(self.subwindow_sizes):
            raise ValueError("signature.subwindow_sizes must be unique.")
        scale_count = len(self.subwindow_sizes) if self.subwindow_sizes else 1
        self.obs_dim = self.logsig_transformer.obs_dim(self.path_builder.base_dim) * scale_count
        if self.add_position:
            self.obs_dim += 1
        self.obs_dim += len(self.account_feature_keys)

    def reset(self, *args, **kwargs) -> Optional[np.ndarray]:
        self.env.reset(*args, **kwargs)
        return self.get_state()

    def _extract_position(self) -> Optional[float]:
        for attr in ("position", "agent_open_position_value"):
            if hasattr(self.env, attr):
                value = getattr(self.env, attr)
                if isinstance(value, torch.Tensor):
                    return float(value.item())
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    def _to_numpy(self, obs: torch.Tensor) -> np.ndarray:
        array = obs.detach().cpu().numpy().astype(np.float32, copy=False)
        array = np.asarray(array, dtype=np.float32).reshape(-1)
        if self.add_position:
            position = self._extract_position()
            if position is not None:
                array = np.concatenate([array, np.array([position], dtype=np.float32)])
        if self.account_feature_keys:
            account_values = self._extract_account_features()
            array = np.concatenate([array, account_values]).astype(np.float32, copy=False)
        return array

    def _extract_account_features(self) -> np.ndarray:
        if not hasattr(self.env, "get_account_features"):
            return np.zeros(len(self.account_feature_keys), dtype=np.float32)
        mapping = self.env.get_account_features()
        values: list[float] = []
        for key in self.account_feature_keys:
            value = mapping.get(key, 0.0)
            values.append(float(value))
        return np.asarray(values, dtype=np.float32)

    def _extract_market_window(self):
        if all(hasattr(self.env, attr) for attr in ("data", "t", "window_size")):
            start = int(self.env.t) - (int(self.env.window_size) - 1)
            end = int(self.env.t) + 1
            window = self.env.data.iloc[start:end]
            if len(window) == int(self.env.window_size):
                return window
        return self.env.get_state()

    def _window_length(self, window) -> int:
        if isinstance(window, pd.DataFrame):
            return int(len(window))
        if isinstance(window, dict):
            if not window:
                return 0
            first = next(iter(window.values()))
            return int(len(first))
        return int(len(window))

    def _slice_window(self, window, size: int):
        if isinstance(window, pd.DataFrame):
            return window.iloc[-size:].reset_index(drop=True)
        if isinstance(window, dict):
            return {key: value[-size:] for key, value in window.items()}
        return window[-size:]

    def _transform_window(self, raw_window) -> torch.Tensor:
        sizes = self.subwindow_sizes or [self._window_length(raw_window)]
        features = []
        for size in sizes:
            if size > self._window_length(raw_window):
                raise ValueError(
                    f"Subwindow size {size} exceeds current window length {self._window_length(raw_window)}."
                )
            path = self.path_builder(self._slice_window(raw_window, size))
            features.append(self.logsig_transformer(path).reshape(-1))
        return torch.cat(features, dim=0) if len(features) > 1 else features[0]

    def get_state(self) -> Optional[np.ndarray]:
        raw = self._extract_market_window()
        if raw is None:
            return None
        logsig = self._transform_window(raw)
        return self._to_numpy(logsig)

    def step(self, action):
        reward, done, _ = self.env.step(action)
        obs = self.get_state()
        return reward, done, obs

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
