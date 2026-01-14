from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.algos.d3qn.trainer import config_from_dict, evaluate, load_config
from rl.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate D3QN trading agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reward", type=str, default=None, choices=["profit", "sr"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    device = args.device or "auto"
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    if "config" in checkpoint:
        config = config_from_dict(checkpoint["config"])
    else:
        config = load_config(Path(args.config))

    if args.device is not None:
        config.run.device = args.device
    if args.reward is not None:
        config.env.reward = args.reward

    epsilon = args.epsilon if args.epsilon is not None else config.train.eval_epsilon

    mean_return, metrics, _ = evaluate(
        config,
        checkpoint_path=checkpoint_path,
        episodes=args.episodes,
        epsilon=epsilon,
        device=config.run.device,
    )
    print(f"Average return over {int(metrics['episodes'])} episodes: {mean_return:.2f}")


if __name__ == "__main__":
    main()
