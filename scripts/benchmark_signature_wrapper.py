from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.algos.d3qn.trainer import _prepare_data, _resolve_obs_dim, load_config
from rl.envs.make_env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark signature observation extraction through SignatureObsWrapper.",
    )
    parser.add_argument("--config", required=True, help="Path to a D3QN config YAML file.")
    parser.add_argument("--num-windows", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _unwrap_env(env):
    current = env
    for _ in range(10):
        if hasattr(current, "_get_valid_start_range"):
            return current
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    return current


def _build_env(config_path: Path, device: str):
    config = load_config(config_path)
    df = _prepare_data(config)
    env = make_env(
        df,
        config.env.reward,
        config.env.window_size,
        device,
        trading_period=config.env.trading_period,
        max_positions=config.env.max_positions,
        max_exposure_ratio=config.env.max_exposure_ratio,
        sell_mode=config.env.sell_mode,
        buy_fractions=config.env.buy_fractions,
        sell_fractions=config.env.sell_fractions,
        action_number=config.agent.action_number,
        action_mode=config.env.action_mode,
        initial_capital=config.env.initial_capital,
        transaction_cost_bps=config.env.transaction_cost_bps,
        slippage_bps=config.env.slippage_bps,
        invalid_sell_penalty=config.env.invalid_sell_penalty,
        blocked_trade_penalty=config.env.blocked_trade_penalty,
        min_hold_steps=config.env.min_hold_steps,
        trade_cooldown_steps=config.env.trade_cooldown_steps,
        dynamic_exposure_enabled=config.env.dynamic_exposure_enabled,
        dynamic_exposure_vol_window=config.env.dynamic_exposure_vol_window,
        dynamic_exposure_min_scale=config.env.dynamic_exposure_min_scale,
        dynamic_exposure_strength=config.env.dynamic_exposure_strength,
        allow_short=config.env.allow_short,
        max_leverage=config.env.max_leverage,
        action_low=config.env.action_low,
        action_high=config.env.action_high,
        min_equity_ratio=config.env.min_equity_ratio,
        stop_on_bankruptcy=config.env.stop_on_bankruptcy,
        obs_config=config.env.obs,
    )
    obs_dim = _resolve_obs_dim(env, config)
    return config, env, obs_dim


def _start_indices(env, count: int) -> list[int]:
    base_env = _unwrap_env(env)
    if not hasattr(base_env, "_get_valid_start_range"):
        raise ValueError("Could not resolve start-index range from environment.")
    min_start, max_start = base_env._get_valid_start_range()
    if max_start < min_start:
        raise ValueError(f"Invalid start-index range [{min_start}, {max_start}].")
    values = np.linspace(min_start, max_start, num=max(count, 1), dtype=int)
    return [int(value) for value in values.tolist()]


def _time_windows(env, start_indices: list[int], expected_obs_dim: int) -> float:
    start = time.perf_counter()
    for start_index in start_indices:
        env.reset(start_index=start_index)
        obs = env.get_state()
        if obs is None:
            raise ValueError(f"Received None observation for start_index={start_index}.")
        if len(obs) != expected_obs_dim:
            raise ValueError(
                f"Observation dimension mismatch for start_index={start_index}: "
                f"expected {expected_obs_dim}, got {len(obs)}."
            )
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    _, env, obs_dim = _build_env(config_path, args.device)
    all_indices = _start_indices(env, args.num_windows + args.warmup)
    warmup_indices = all_indices[: args.warmup]
    measure_indices = all_indices[args.warmup :]
    if not measure_indices:
        raise ValueError("num-windows must be >= 1 after warmup is removed.")

    for start_index in warmup_indices:
        env.reset(start_index=start_index)
        obs = env.get_state()
        if obs is None:
            raise ValueError(f"Warmup observation is None for start_index={start_index}.")

    elapsed_values: list[float] = []
    for _ in range(args.repeats):
        elapsed_values.append(_time_windows(env, measure_indices, obs_dim))

    elapsed_mean = float(np.mean(elapsed_values))
    elapsed_std = float(np.std(elapsed_values, ddof=0))
    windows = len(measure_indices)
    print(f"config={args.config}")
    print(f"windows={windows}")
    print(f"warmup={args.warmup}")
    print(f"repeats={args.repeats}")
    print(f"obs_dim={obs_dim}")
    print(f"elapsed_mean_sec={elapsed_mean:.6f}")
    print(f"elapsed_std_sec={elapsed_std:.6f}")
    print(f"ms_per_window_mean={elapsed_mean * 1000.0 / max(windows, 1):.4f}")
    print(f"windows_per_sec_mean={windows / max(elapsed_mean, 1e-12):.2f}")


if __name__ == "__main__":
    main()
