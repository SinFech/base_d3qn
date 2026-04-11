from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rl.envs.make_env import load_price_data
from rl.envs.wrappers import SignatureObsWrapper
from rl.features.signature import LogSigTransformer, PathBuilder


def test_load_price_data_parses_ohlcv_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date,Price,Open,High,Low,Vol.,Change %",
                "\"Jan 02, 2024\",\"1,234.5\",\"1,200.0\",\"1,260.0\",\"1,180.0\",12.5K,2.50%",
                "\"Jan 01, 2024\",\"1,000.0\",990.0,\"1,010.0\",980.0,1.25M,-1.00%",
            ]
        )
    )

    df = load_price_data(csv_path)

    assert df["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-02"]
    assert df["Close"].tolist() == [1000.0, 1234.5]
    assert df["Open"].tolist() == [990.0, 1200.0]
    assert df["High"].tolist() == [1010.0, 1260.0]
    assert df["Low"].tolist() == [980.0, 1180.0]
    assert df["Volume"].tolist() == [1_250_000.0, 12_500.0]
    assert df["ChangePct"].tolist() == [-0.01, 0.025]


def test_path_builder_supports_standardization_basepoint_and_extra_channels() -> None:
    window = pd.DataFrame(
        {
            "Close": [100.0, 104.0, 102.0, 109.0, 111.0],
            "High": [101.0, 106.0, 103.0, 111.0, 112.0],
            "Low": [99.0, 101.0, 100.0, 107.0, 110.0],
            "Volume": [10.0, 20.0, 30.0, 25.0, 15.0],
        }
    )
    builder = PathBuilder(
        embedding={
            "log_price": {},
            "normalized_cumulative_volume": {},
            "high_low_range": {},
        },
        standardize_path_channels=True,
        basepoint=True,
    )

    path = builder.build(window)

    assert tuple(path.shape) == (len(window) + 1, 3)
    assert torch.allclose(path[0], torch.zeros(3, dtype=path.dtype))
    assert path[-1, 1].item() > 0.0
    assert path[-1, 2].item() > 0.0

    price_channel = path[1:, 0]
    diff_std = torch.std(price_channel[1:] - price_channel[:-1], unbiased=False)
    assert diff_std.item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


class _DummyEnv:
    def __init__(self, data: pd.DataFrame, window_size: int = 5) -> None:
        self.data = data
        self.window_size = window_size
        self.t = window_size - 1

    def reset(self, *args, **kwargs) -> None:
        self.t = self.window_size - 1

    def get_state(self) -> torch.Tensor:
        window = self.data.iloc[self.t - (self.window_size - 1) : self.t + 1]["Close"]
        return torch.tensor(window.to_numpy(), dtype=torch.float32)


def test_signature_wrapper_multiscale_obs_dim_matches_output() -> None:
    data = pd.DataFrame(
        {
            "Close": np.linspace(100.0, 110.0, 6),
            "High": np.linspace(101.0, 111.0, 6),
            "Low": np.linspace(99.0, 109.0, 6),
            "Volume": np.linspace(10.0, 60.0, 6),
        }
    )
    path_builder = PathBuilder(
        embedding={
            "log_price": {},
            "normalized_cumulative_volume": {},
        },
        basepoint=True,
    )
    transformer = LogSigTransformer(
        degree=2,
        method=1,
        time_aug=False,
        lead_lag=False,
        n_jobs=1,
        use_disk_prepare_cache=False,
        base_dim=path_builder.base_dim,
    )
    wrapper = SignatureObsWrapper(
        env=_DummyEnv(data, window_size=5),
        path_builder=path_builder,
        logsig_transformer=transformer,
        subwindow_sizes=[3, 5],
    )

    obs = wrapper.get_state()

    assert obs is not None
    assert obs.shape == (wrapper.obs_dim,)
