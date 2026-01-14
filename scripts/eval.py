from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.algos.d3qn.trainer import Config, build_agent, config_from_dict, load_config
from rl.envs.make_env import filter_date_range, load_price_data, make_env
from rl.utils.checkpoint import load_checkpoint
from rl.utils.logging import setup_run_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate D3QN trading agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reward", type=str, default=None, choices=["profit", "sr"])
    return parser.parse_args()


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _prepare_data(config: Config):
    df = load_price_data(
        Path(config.data.path),
        price_column=config.data.price_column,
        close_column=config.data.close_column,
        date_column=config.data.date_column,
    )
    df = filter_date_range(
        df,
        date_column=config.data.date_column,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
    )
    return df


def _resolve_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _compute_start_range(
    data_len: int,
    window_size: int,
    trading_period: Optional[int],
) -> tuple[int, int]:
    min_start = window_size - 1
    max_start = data_len - 1
    if trading_period is not None:
        max_start = data_len - trading_period
    return min_start, max_start


def _sample_start_indices(
    rng: np.random.Generator,
    min_start: int,
    max_start: int,
    num_episodes: int,
    replace: bool,
) -> List[int]:
    choices = np.arange(min_start, max_start + 1)
    picks = rng.choice(choices, size=num_episodes, replace=replace)
    return [int(value) for value in picks.tolist()]


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    run_dir = _resolve_run_dir(checkpoint_path)
    logger = setup_run_logger("eval", run_dir)

    device_override = args.device or "auto"
    checkpoint = load_checkpoint(checkpoint_path, device=device_override)
    if "config" in checkpoint:
        config = config_from_dict(checkpoint["config"])
    else:
        config = load_config(Path(args.config))

    if args.device is not None:
        config.run.device = args.device
    if args.reward is not None:
        config.env.reward = args.reward

    eval_cfg = config.eval
    if args.episodes is not None:
        eval_cfg.num_episodes = args.episodes
    if args.epsilon is not None:
        eval_cfg.epsilon = args.epsilon

    eval_seed = eval_cfg.seed
    fixed_windows_seed = eval_cfg.fixed_windows_seed if eval_cfg.fixed_windows_seed is not None else eval_seed

    set_global_seeds(eval_seed)
    device = _resolve_device(config.run.device)

    df = _prepare_data(config)
    data_len = len(df)
    min_start, max_start = _compute_start_range(
        data_len,
        config.env.window_size,
        config.env.trading_period,
    )
    if max_start < min_start:
        raise ValueError(
            "Invalid start_index range "
            f"[{min_start}, {max_start}] for data length {data_len} "
            f"and trading_period {config.env.trading_period}."
        )

    if eval_cfg.fixed_windows:
        rng = np.random.default_rng(fixed_windows_seed)
        available = max_start - min_start + 1
        replace = available < eval_cfg.num_episodes
        if replace:
            logger.warning("Not enough unique start windows; sampling with replacement.")
        start_indices = _sample_start_indices(
            rng,
            min_start,
            max_start,
            eval_cfg.num_episodes,
            replace=replace,
        )
    else:
        rng = np.random.default_rng(eval_seed)
        start_indices = _sample_start_indices(
            rng,
            min_start,
            max_start,
            eval_cfg.num_episodes,
            replace=True,
        )

    env = make_env(
        df,
        config.env.reward,
        config.env.window_size,
        device,
        trading_period=config.env.trading_period,
        obs_config=config.env.obs,
    )
    obs_dim = getattr(env, "obs_dim", config.env.window_size)
    config.agent.input_dim = obs_dim
    agent = build_agent(config, device, input_dim=obs_dim)
    agent.policy_net.load_state_dict(checkpoint["policy_state"])
    agent.target_net.load_state_dict(checkpoint["target_state"])
    agent.policy_net.eval()
    agent.target_net.eval()

    returns: List[float] = []
    episode_rows: List[dict] = []
    with torch.no_grad():
        for episode_id, start_index in enumerate(start_indices):
            env.reset(seed=eval_seed, start_index=start_index)
            agent.reset_episode()
            state = env.get_state()
            episode_return = 0.0

            while state is not None:
                action = agent.select_action(
                    state,
                    training=False,
                    epsilon_override=eval_cfg.epsilon,
                )
                reward, done, _ = env.step(action)
                episode_return += reward.item()
                state = env.get_state()
                if done:
                    break

            returns.append(episode_return)
            row = {
                "episode_id": episode_id,
                "start_index": int(env.last_start_index)
                if env.last_start_index is not None
                else int(start_index),
                "episode_return": episode_return,
            }
            if env.last_start_timestamp is not None:
                row["start_timestamp"] = env.last_start_timestamp
            episode_rows.append(row)

    include_timestamp = any("start_timestamp" in row for row in episode_rows)
    if include_timestamp:
        for row in episode_rows:
            row.setdefault("start_timestamp", "")

    run_dir.mkdir(parents=True, exist_ok=True)

    if eval_cfg.save_per_episode:
        episodes_path = run_dir / "eval_episodes.csv"
        fieldnames = ["episode_id", "start_index"]
        if include_timestamp:
            fieldnames.append("start_timestamp")
        fieldnames.append("episode_return")
        with episodes_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in episode_rows:
                writer.writerow(row)

    returns_array = np.array(returns, dtype=float)
    mean_return = float(np.mean(returns_array)) if returns_array.size else 0.0
    std_return = float(np.std(returns_array)) if returns_array.size else 0.0
    median_return = float(np.median(returns_array)) if returns_array.size else 0.0
    min_return = float(np.min(returns_array)) if returns_array.size else 0.0
    max_return = float(np.max(returns_array)) if returns_array.size else 0.0
    p25 = float(np.percentile(returns_array, 25)) if returns_array.size else 0.0
    p75 = float(np.percentile(returns_array, 75)) if returns_array.size else 0.0

    summary = {
        "num_episodes": eval_cfg.num_episodes,
        "mean_return": mean_return,
        "std_return": std_return,
        "median_return": median_return,
        "min_return": min_return,
        "max_return": max_return,
        "p25": p25,
        "p75": p75,
        "eval_config": {
            "seed": eval_seed,
            "fixed_windows": eval_cfg.fixed_windows,
            "fixed_windows_seed": fixed_windows_seed,
            "epsilon": eval_cfg.epsilon,
            "save_per_episode": eval_cfg.save_per_episode,
        },
        "window_size": config.env.window_size,
        "trading_period": config.env.trading_period,
        "data_range": {
            "start_date": config.data.start_date,
            "end_date": config.data.end_date,
        },
        "data_len": data_len,
        "start_indices": start_indices,
    }

    if include_timestamp:
        summary["start_timestamps"] = [row["start_timestamp"] for row in episode_rows]

    summary_path = run_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Evaluation average return over %s episodes: %.2f",
        eval_cfg.num_episodes,
        mean_return,
    )


if __name__ == "__main__":
    main()
