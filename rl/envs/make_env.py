from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from rl.envs.trading_env import TradingEnvironment
from rl.envs.trading_env_continuous import ContinuousTradingEnvironment
from rl.envs.trading_env_discrete_capital import DiscreteCapitalTradingEnvironment


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
    max_exposure_ratio: Optional[float] = 1.0,
    sell_mode: str = "all",
    buy_fractions: Optional[list[float]] = None,
    sell_fractions: Optional[list[float]] = None,
    action_number: Optional[int] = None,
    action_mode: str = "discrete",
    initial_capital: float = 100_000.0,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 2.0,
    invalid_sell_penalty: float = 0.1,
    allow_short: bool = False,
    max_leverage: float = 1.0,
    action_low: float = 0.0,
    action_high: float = 1.0,
    min_equity_ratio: float = 0.2,
    stop_on_bankruptcy: bool = True,
    obs_config=None,
) -> object:
    if action_mode == "continuous":
        env = ContinuousTradingEnvironment(
            df,
            reward=reward,
            window_size=window_size,
            trading_period=trading_period,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            allow_short=allow_short,
            max_leverage=max_leverage,
            action_low=action_low,
            action_high=action_high,
            min_equity_ratio=min_equity_ratio,
            stop_on_bankruptcy=stop_on_bankruptcy,
            device=device,
        )
    elif action_mode == "discrete_capital":
        env = DiscreteCapitalTradingEnvironment(
            df,
            reward=reward,
            window_size=window_size,
            trading_period=trading_period,
            max_positions=max_positions,
            max_exposure_ratio=max_exposure_ratio,
            sell_mode=sell_mode,
            buy_fractions=buy_fractions,
            sell_fractions=sell_fractions,
            action_number=action_number,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            invalid_sell_penalty=invalid_sell_penalty,
            min_equity_ratio=min_equity_ratio,
            stop_on_bankruptcy=stop_on_bankruptcy,
            device=device,
        )
    elif action_mode == "discrete":
        env = TradingEnvironment(
            df,
            reward=reward,
            window_size=window_size,
            trading_period=trading_period,
            device=device,
            max_positions=max_positions,
            sell_mode=sell_mode,
        )
    else:
        raise ValueError("action_mode must be 'discrete', 'discrete_capital', or 'continuous'.")

    obs_type = _get_cfg_value(obs_config, "type", "raw")
    if obs_type == "signature":
        signature_cfg = _get_cfg_value(obs_config, "signature", {})
        logsig_cfg = _get_cfg_value(signature_cfg, "logsig", {})
        torch_cfg = _get_cfg_value(signature_cfg, "torch", {})
        perf_cfg = _get_cfg_value(signature_cfg, "perf", {})
        account_feature_keys = _get_cfg_value(signature_cfg, "account_features", None)

        from rl.envs.wrappers import SignatureObsWrapper
        from rl.features.signature import LogSigTransformer, PathBuilder

        backend = _get_cfg_value(signature_cfg, "backend", "pysiglib")
        if backend != "pysiglib":
            raise ValueError(f"Unsupported signature backend: {backend}")
        path_builder = PathBuilder(
            embedding=_get_cfg_value(signature_cfg, "embedding", "price_return"),
            rolling_mean_window=_get_cfg_value(signature_cfg, "rolling_mean_window", 5),
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
        env = SignatureObsWrapper(
            env,
            path_builder,
            transformer,
            add_position=False,
            account_feature_keys=account_feature_keys,
        )
    elif obs_type == "raw":
        env.obs_dim = window_size
    else:
        raise ValueError(f"Unsupported obs.type: {obs_type}")
    return env
