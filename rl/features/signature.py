from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
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
        }
        normalized: list[str] = []
        for raw in components:
            if not isinstance(raw, str):
                raise ValueError("embedding components must be strings.")
            key = raw.strip()
            key = alias_map.get(key, key)
            if key not in {"log_price", "log_return", "rolling_mean", "rolling_vol"}:
                raise ValueError(f"Unsupported embedding component: {raw}")
            normalized.append(key)

        if len(set(normalized)) != len(normalized):
            raise ValueError("embedding components must be unique.")
        normalized_options = {alias_map.get(key, key): value for key, value in options.items()}
        return normalized, normalized_options

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

    def build(self, window_close: np.ndarray | torch.Tensor) -> torch.Tensor:
        close = torch.as_tensor(window_close, device=self.device, dtype=self.dtype)
        if close.ndim == 2 and close.shape[1] == 1:
            close = close.squeeze(1)
        if close.ndim != 1:
            raise ValueError("window_close must have shape (W,) or (W, 1)")

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
        return torch.stack(features, dim=-1)

    def __call__(self, window_close: np.ndarray | torch.Tensor) -> torch.Tensor:
        return self.build(window_close)


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
