from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.algos.d3qn.trainer import Config, load_config, train
from rl.utils.path import build_run_name, build_run_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train D3QN trading agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--reward", type=str, default=None, choices=["profit", "sr"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trading-period", type=int, default=None)
    parser.add_argument("--train-split", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    if args.seed is not None:
        config.run.seed = args.seed
    if args.num_episodes is not None:
        config.train.num_episodes = args.num_episodes
    if args.max_steps_per_episode is not None:
        config.train.max_steps_per_episode = args.max_steps_per_episode
    if args.total_steps is not None:
        config.train.max_total_steps = args.total_steps
    if args.reward is not None:
        config.env.reward = args.reward
    if args.device is not None:
        config.run.device = args.device
    if args.trading_period is not None:
        config.env.trading_period = args.trading_period
    if args.train_split is not None:
        config.env.train_split = args.train_split
    if args.log_interval is not None:
        config.train.log_interval = args.log_interval
    if args.checkpoint_interval is not None:
        config.train.checkpoint_interval = args.checkpoint_interval
    return config


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    config = apply_overrides(config, args)

    run_name = args.run_name or build_run_name(None)
    run_paths = build_run_paths(Path(args.output_dir), run_name)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)

    train(config, run_paths)
    print(f"Run artifacts saved to {run_paths.run_dir}")


if __name__ == "__main__":
    main()
