from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.envs.make_env import filter_date_range, load_price_data
from rl.features.signature import LogSigTransformer, PathBuilder


def _get_nested(mapping: dict, *keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark signature feature extraction.")
    parser.add_argument("--config", required=True, help="Path to a config YAML file.")
    parser.add_argument("--num-windows", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=32)
    return parser.parse_args()


def _load_windows(config: dict, num_windows: int) -> tuple[list, int]:
    data_cfg = config.get("data", {}) or {}
    env_cfg = config.get("env", {}) or {}
    window_size = int(env_cfg.get("window_size", 24))

    df = load_price_data(
        Path(data_cfg.get("path", "data/Bitcoin History 2010-2024.csv")),
        price_column=data_cfg.get("price_column", "Price"),
        close_column=data_cfg.get("close_column", "Close"),
        date_column=data_cfg.get("date_column", "Date"),
    )
    df = filter_date_range(
        df,
        date_column=data_cfg.get("date_column", "Date"),
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
    )
    min_end = window_size - 1
    max_end = len(df) - 1
    indices = np.linspace(min_end, max_end, num=max(num_windows, 1), dtype=int)
    windows = [df.iloc[end - window_size + 1 : end + 1].reset_index(drop=True) for end in indices]
    return windows, window_size


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text()) or {}
    signature_cfg = _get_nested(raw, "env", "obs", "signature", default={}) or {}
    logsig_cfg = signature_cfg.get("logsig", {}) or {}
    torch_cfg = signature_cfg.get("torch", {}) or {}
    perf_cfg = signature_cfg.get("perf", {}) or {}

    builder = PathBuilder(
        embedding=signature_cfg.get("embedding", {"log_price": {}, "log_return": {}}),
        rolling_mean_window=signature_cfg.get("rolling_mean_window", 5),
        standardize_path_channels=signature_cfg.get("standardize_path_channels", False),
        basepoint=signature_cfg.get("basepoint", False),
        device=torch_cfg.get("device", "cpu"),
        dtype=torch_cfg.get("dtype", "float32"),
    )
    transformer = LogSigTransformer(
        degree=logsig_cfg.get("degree", 2),
        method=logsig_cfg.get("method", 1),
        time_aug=logsig_cfg.get("time_aug", True),
        lead_lag=logsig_cfg.get("lead_lag", False),
        end_time=logsig_cfg.get("end_time", 1.0),
        n_jobs=perf_cfg.get("n_jobs", -1),
        device=torch_cfg.get("device", "cpu"),
        dtype=torch_cfg.get("dtype", "float32"),
        use_disk_prepare_cache=perf_cfg.get("use_disk_prepare_cache", False),
        prepare_cache_dir=perf_cfg.get("prepare_cache_dir", "data/pysiglib_prepare_cache"),
        base_dim=builder.base_dim,
    )

    windows, window_size = _load_windows(raw, args.num_windows)
    subwindow_sizes = [int(size) for size in signature_cfg.get("subwindow_sizes", [])]
    sizes = subwindow_sizes or [window_size]

    def compute_feature(window):
        outputs = []
        for size in sizes:
            sliced = window.iloc[-size:].reset_index(drop=True)
            outputs.append(transformer(builder(sliced)).reshape(-1))
        merged = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        return merged.detach().cpu().numpy()

    for window in windows[: args.warmup]:
        compute_feature(window)

    start = time.perf_counter()
    features = [compute_feature(window) for window in windows]
    elapsed = time.perf_counter() - start

    obs_dim = int(np.asarray(features[0]).reshape(-1).shape[0])
    count = len(features)
    print(f"config={args.config}")
    print(f"windows={count}")
    print(f"subwindow_sizes={sizes}")
    print(f"obs_dim={obs_dim}")
    print(f"elapsed_sec={elapsed:.6f}")
    print(f"ms_per_window={elapsed * 1000.0 / max(count, 1):.4f}")
    print(f"windows_per_sec={count / max(elapsed, 1e-12):.2f}")


if __name__ == "__main__":
    main()
