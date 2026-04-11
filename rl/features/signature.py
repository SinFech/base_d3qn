from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch
import pysiglib  # type: ignore


def _resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.lower()
    if normalized in {"float32", "float", "fp32"}:
        return torch.float32
    if normalized in {"float64", "double", "fp64"}:
        return torch.float64
    raise ValueError(f"Unsupported torch dtype: {dtype}")


class PathBuilder:
    """Builds an embedded path tensor for signature computation."""

    def __init__(
        self,
        embedding: Optional[dict] = None,
        rolling_mean_window: int = 5,
        standardize_path_channels: bool = False,
        basepoint: bool = False,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype = "float32",
        eps: float = 1e-8,
    ) -> None:
        if embedding is None:
            embedding = {"log_price": {}, "log_return": {}}
        self.embedding = embedding
        self.device = torch.device(device)
        self.dtype = _resolve_torch_dtype(dtype)
        self.eps = eps
        self.standardize_path_channels = bool(standardize_path_channels)
        self.basepoint = bool(basepoint)
        if rolling_mean_window < 1:
            raise ValueError("rolling_mean_window must be >= 1.")
        self.rolling_mean_window = int(rolling_mean_window)
        self.embedding_components, self.embedding_options = self._resolve_embedding_components(embedding)
        self.base_dim = len(self.embedding_components)

    def _resolve_embedding_components(self, embedding: dict) -> tuple[list[str], dict[str, dict]]:
        if not isinstance(embedding, dict):
            raise ValueError("embedding must be a non-empty dict.")
        if not embedding:
            raise ValueError("embedding must be a non-empty dict.")
        options: dict[str, dict] = {}
        components = list(embedding.keys())
        for key, value in embedding.items():
            if value is None:
                options[key] = {}
            elif isinstance(value, dict):
                options[key] = value
            else:
                raise ValueError("embedding component config must be a dict.")

        alias_map = {
            "price": "log_price",
            "logprice": "log_price",
            "log_price": "log_price",
            "return": "log_return",
            "logreturn": "log_return",
            "log_return": "log_return",
            "rolling_mean": "rolling_mean",
            "rolling_vol": "rolling_vol",
            "rolling_std": "rolling_vol",
            "vol": "rolling_vol",
            "volatility": "rolling_vol",
            "normalized_cumulative_volume": "normalized_cumulative_volume",
            "cumulative_volume": "normalized_cumulative_volume",
            "cum_volume": "normalized_cumulative_volume",
            "volume_profile": "normalized_cumulative_volume",
            "high_low_range": "high_low_range",
            "hl_range": "high_low_range",
            "range_proxy": "high_low_range",
        }
        normalized: list[str] = []
        for raw in components:
            if not isinstance(raw, str):
                raise ValueError("embedding components must be strings.")
            key = raw.strip()
            key = alias_map.get(key, key)
            if key not in {
                "log_price",
                "log_return",
                "rolling_mean",
                "rolling_vol",
                "normalized_cumulative_volume",
                "high_low_range",
            }:
                raise ValueError(f"Unsupported embedding component: {raw}")
            normalized.append(key)

        if len(set(normalized)) != len(normalized):
            raise ValueError("embedding components must be unique.")
        normalized_options = {alias_map.get(key, key): value for key, value in options.items()}
        return normalized, normalized_options

    def _to_1d_tensor(self, values) -> torch.Tensor:
        tensor = torch.as_tensor(values, device=self.device, dtype=self.dtype)
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim != 1:
            raise ValueError("PathBuilder inputs must resolve to 1D channels.")
        return tensor

    def _normalize_input_key(self, key: str) -> str:
        text = key.strip().lower().replace(".", "").replace("%", "pct")
        text = text.replace(" ", "_")
        alias_map = {
            "price": "close",
            "close": "close",
            "open": "open",
            "high": "high",
            "low": "low",
            "vol": "volume",
            "volume": "volume",
            "changepct": "changepct",
            "change_pct": "changepct",
        }
        return alias_map.get(text, text)

    def _resolve_input_channels(
        self,
        window_data: np.ndarray | torch.Tensor | pd.DataFrame | Mapping[str, np.ndarray | torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if isinstance(window_data, pd.DataFrame):
            channels: dict[str, torch.Tensor] = {}
            for column in window_data.columns:
                if not pd.api.types.is_numeric_dtype(window_data[column]):
                    continue
                channels[self._normalize_input_key(str(column))] = self._to_1d_tensor(
                    window_data[column].to_numpy()
                )
            return channels
        if isinstance(window_data, Mapping):
            return {
                self._normalize_input_key(str(key)): self._to_1d_tensor(value)
                for key, value in window_data.items()
            }
        close = self._to_1d_tensor(window_data)
        return {"close": close}

    def _require_channel(self, channels: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        if name not in channels:
            raise ValueError(f"Embedding component requires '{name}' data in the current window.")
        return channels[name]

    def _rolling_mean(self, values: torch.Tensor, window: Optional[int] = None) -> torch.Tensor:
        resolved_window = self.rolling_mean_window if window is None else int(window)
        if resolved_window <= 1:
            return values.clone()
        cumsum = torch.cumsum(values, dim=0)
        idx = torch.arange(values.shape[0], device=values.device)
        start = idx - (resolved_window - 1)
        start = torch.clamp(start, min=0)
        prev = torch.where(start > 0, cumsum[start - 1], torch.zeros_like(cumsum))
        window_sum = cumsum - prev
        denom = torch.minimum(idx + 1, torch.full_like(idx, resolved_window)).to(values.dtype)
        return window_sum / denom

    def _rolling_std(self, values: torch.Tensor, window: Optional[int] = None) -> torch.Tensor:
        resolved_window = self.rolling_mean_window if window is None else int(window)
        if resolved_window <= 1:
            return torch.zeros_like(values)
        cumsum = torch.cumsum(values, dim=0)
        cumsum_sq = torch.cumsum(values * values, dim=0)
        idx = torch.arange(values.shape[0], device=values.device)
        start = idx - (resolved_window - 1)
        start = torch.clamp(start, min=0)
        prev = torch.where(start > 0, cumsum[start - 1], torch.zeros_like(cumsum))
        prev_sq = torch.where(start > 0, cumsum_sq[start - 1], torch.zeros_like(cumsum_sq))
        window_sum = cumsum - prev
        window_sum_sq = cumsum_sq - prev_sq
        denom = torch.minimum(idx + 1, torch.full_like(idx, resolved_window)).to(values.dtype)
        mean = window_sum / denom
        mean_sq = window_sum_sq / denom
        var = torch.clamp(mean_sq - mean * mean, min=0.0)
        return torch.sqrt(var)

    def _standardize_channel(self, channel: torch.Tensor) -> torch.Tensor:
        if channel.numel() <= 1:
            return channel.clone()
        deltas = channel[1:] - channel[:-1]
        scale = torch.std(deltas, unbiased=False)
        if not torch.isfinite(scale) or float(scale) <= self.eps:
            return channel.clone()
        return channel / scale

    def build(
        self,
        window_data: np.ndarray | torch.Tensor | pd.DataFrame | Mapping[str, np.ndarray | torch.Tensor],
    ) -> torch.Tensor:
        channels = self._resolve_input_channels(window_data)
        close = self._require_channel(channels, "close")
        close = torch.clamp(close, min=self.eps)
        log_close = torch.log(close)
        rel_logprice = log_close - log_close[0]
        logreturn = torch.zeros_like(log_close)
        if log_close.numel() > 1:
            logreturn[1:] = log_close[1:] - log_close[:-1]
        features = []
        for component in self.embedding_components:
            if component == "log_price":
                features.append(rel_logprice)
            elif component == "log_return":
                features.append(logreturn)
            elif component == "rolling_mean":
                window = self.embedding_options.get("rolling_mean", {}).get("window", self.rolling_mean_window)
                features.append(self._rolling_mean(logreturn, window=window))
            elif component == "rolling_vol":
                window = self.embedding_options.get("rolling_vol", {}).get("window", self.rolling_mean_window)
                features.append(self._rolling_std(logreturn, window=window))
            elif component == "normalized_cumulative_volume":
                volume = torch.clamp(self._require_channel(channels, "volume"), min=0.0)
                total_volume = torch.clamp(volume.sum(), min=self.eps)
                features.append(torch.cumsum(volume, dim=0) / total_volume)
            elif component == "high_low_range":
                high = torch.clamp(self._require_channel(channels, "high"), min=self.eps)
                low = torch.clamp(self._require_channel(channels, "low"), min=self.eps)
                features.append((high - low) / close)
        if self.standardize_path_channels:
            features = [self._standardize_channel(feature) for feature in features]
        path = torch.stack(features, dim=-1)
        if self.basepoint:
            zero_row = torch.zeros((1, path.shape[1]), device=path.device, dtype=path.dtype)
            path = torch.cat([zero_row, path], dim=0)
        return path

    def __call__(
        self,
        window_data: np.ndarray | torch.Tensor | pd.DataFrame | Mapping[str, np.ndarray | torch.Tensor],
    ) -> torch.Tensor:
        return self.build(window_data)


class LogSigTransformer:
    """Computes logsignature features using pysiglib."""

    def __init__(
        self,
        degree: int = 2,
        method: int = 1,
        time_aug: bool = True,
        lead_lag: bool = False,
        end_time: float = 1.0,
        n_jobs: int = -1,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype = "float32",
        use_disk_prepare_cache: bool = True,
        prepare_cache_dir: str | Path = "data/pysiglib_prepare_cache",
        base_dim: Optional[int] = None,
    ) -> None:
        self.degree = int(degree)
        self.method = int(method)
        self.time_aug = bool(time_aug)
        self.lead_lag = bool(lead_lag)
        self.end_time = float(end_time)
        self.n_jobs = int(n_jobs)
        self.device = torch.device(device)
        self.dtype = _resolve_torch_dtype(dtype)
        self.use_disk_prepare_cache = bool(use_disk_prepare_cache)
        self.prepare_cache_dir = Path(prepare_cache_dir)

        if self.use_disk_prepare_cache:
            self.prepare_cache_dir.mkdir(parents=True, exist_ok=True)

        if base_dim is not None and hasattr(pysiglib, "prepare_log_sig"):
            pysiglib.prepare_log_sig(
                base_dim,
                self.degree,
                self.method,
                self.time_aug,
                self.lead_lag,
                self.use_disk_prepare_cache,
            )

    def obs_dim(self, base_dim: int) -> int:
        return int(pysiglib.log_sig_length(base_dim, self.degree, self.time_aug, self.lead_lag))

    def transform(self, path: torch.Tensor) -> torch.Tensor:
        if path.ndim == 2:
            return self._transform_single(path)
        if path.ndim == 3:
            return torch.stack([self._transform_single(sample) for sample in path], dim=0)
        raise ValueError("path must have shape (W, D) or (B, W, D)")

    def _transform_single(self, path: torch.Tensor) -> torch.Tensor:
        path = path.to(device=self.device, dtype=self.dtype)
        return pysiglib.log_sig(
            path,
            self.degree,
            self.time_aug,
            self.lead_lag,
            self.end_time,
            self.method,
            self.n_jobs,
        )

    def __call__(self, path: torch.Tensor) -> torch.Tensor:
        return self.transform(path)
