"""Environment wrappers for trading environments."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
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
    ) -> None:
        self.env = env
        self.path_builder = path_builder
        self.logsig_transformer = logsig_transformer
        self.add_position = add_position
        self.account_feature_keys = list(account_feature_keys or [])
        self.obs_dim = self.logsig_transformer.obs_dim(self.path_builder.base_dim)
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

    def get_state(self) -> Optional[np.ndarray]:
        raw = self.env.get_state()
        if raw is None:
            return None
        path = self.path_builder(raw)
        logsig = self.logsig_transformer(path)
        return self._to_numpy(logsig)

    def step(self, action):
        reward, done, _ = self.env.step(action)
        obs = self.get_state()
        return reward, done, obs

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
