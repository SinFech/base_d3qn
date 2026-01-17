from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from rl.envs.trading_env import TradingEnvironment


def _get_cfg_value(config, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def load_price_data(
    path: Path,
    price_column: str = "Price",
    close_column: str = "Close",
    date_column: str = "Date",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={price_column: close_column})
    df[close_column] = df[close_column].astype(str).str.replace(",", "").astype(float)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column).reset_index(drop=True)
    return df


def filter_date_range(
    df: pd.DataFrame,
    date_column: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if start_date is None and end_date is None:
        return df
    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        mask &= df[date_column] >= start_date
    if end_date is not None:
        mask &= df[date_column] <= end_date
    return df.loc[mask].reset_index(drop=True)


def sample_train_test_split(
    df: pd.DataFrame,
    trading_period: int,
    train_split: float,
    index: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= trading_period + 1:
        raise ValueError("Dataframe is too short for the requested trading period.")
    if index is None:
        import random

        index = random.randrange(len(df) - trading_period - 1)
    train_size = int(trading_period * train_split)
    train_df = df[index : index + train_size]
    test_df = df[index + train_size : index + trading_period]
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_env(
    df: pd.DataFrame,
    reward: str,
    window_size: int,
    device: str,
    trading_period: Optional[int] = None,
    max_positions: Optional[int] = None,
    sell_mode: str = "all",
    obs_config=None,
) -> TradingEnvironment:
    env = TradingEnvironment(
        df,
        reward=reward,
        window_size=window_size,
        trading_period=trading_period,
        device=device,
        max_positions=max_positions,
        sell_mode=sell_mode,
    )
    obs_type = _get_cfg_value(obs_config, "type", "raw")
    if obs_type == "signature":
        signature_cfg = _get_cfg_value(obs_config, "signature", {})
        logsig_cfg = _get_cfg_value(signature_cfg, "logsig", {})
        torch_cfg = _get_cfg_value(signature_cfg, "torch", {})
        perf_cfg = _get_cfg_value(signature_cfg, "perf", {})

        from rl.envs.wrappers import SignatureObsWrapper
        from rl.features.signature import LogSigTransformer, PathBuilder

        backend = _get_cfg_value(signature_cfg, "backend", "pysiglib")
        if backend != "pysiglib":
            raise ValueError(f"Unsupported signature backend: {backend}")
        path_builder = PathBuilder(
            embedding=_get_cfg_value(signature_cfg, "embedding", "price_return"),
            device=_get_cfg_value(torch_cfg, "device", "cpu"),
            dtype=_get_cfg_value(torch_cfg, "dtype", "float32"),
        )
        transformer = LogSigTransformer(
            degree=_get_cfg_value(logsig_cfg, "degree", 2),
            method=_get_cfg_value(logsig_cfg, "method", 1),
            time_aug=_get_cfg_value(logsig_cfg, "time_aug", True),
            lead_lag=_get_cfg_value(logsig_cfg, "lead_lag", False),
            end_time=_get_cfg_value(logsig_cfg, "end_time", 1.0),
            n_jobs=_get_cfg_value(perf_cfg, "n_jobs", -1),
            device=_get_cfg_value(torch_cfg, "device", "cpu"),
            dtype=_get_cfg_value(torch_cfg, "dtype", "float32"),
            use_disk_prepare_cache=_get_cfg_value(perf_cfg, "use_disk_prepare_cache", True),
            prepare_cache_dir=_get_cfg_value(perf_cfg, "prepare_cache_dir", "data/pysiglib_prepare_cache"),
            base_dim=path_builder.base_dim,
        )
        env = SignatureObsWrapper(env, path_builder, transformer, add_position=False)
    elif obs_type == "raw":
        env.obs_dim = window_size
    else:
        raise ValueError(f"Unsupported obs.type: {obs_type}")
    return env
