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
        embedding: str = "price_return",
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype = "float32",
        eps: float = 1e-8,
    ) -> None:
        if embedding != "price_return":
            raise ValueError(f"Unsupported embedding: {embedding}")
        self.embedding = embedding
        self.device = torch.device(device)
        self.dtype = _resolve_torch_dtype(dtype)
        self.eps = eps
        self.base_dim = 2

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
        return torch.stack([rel_logprice, logreturn], dim=-1)

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
