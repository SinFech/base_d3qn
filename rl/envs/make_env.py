from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from rl.envs.trading_env import TradingEnvironment


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
) -> TradingEnvironment:
    return TradingEnvironment(
        df,
        reward=reward,
        window_size=window_size,
        trading_period=trading_period,
        device=device,
    )
