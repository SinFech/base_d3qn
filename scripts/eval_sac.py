from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.algos.sac.trainer import config_from_dict, evaluate, load_config
from rl.utils.checkpoint import load_checkpoint
from rl.utils.logging import setup_run_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC trading agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/sac_signature.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reward", type=str, default=None, choices=["profit", "sr", "sr_enhanced"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--trading-period", type=int, default=None)
    parser.add_argument("--initial-capital", type=float, default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--action-low", type=float, default=None)
    parser.add_argument("--action-high", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _resolve_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    run_dir = Path(args.output_dir) if args.output_dir else _resolve_run_dir(checkpoint_path)
    logger = setup_run_logger("eval_sac", run_dir)

    checkpoint = load_checkpoint(checkpoint_path, device=args.device or "auto")
    if "config" in checkpoint:
        config = config_from_dict(checkpoint["config"])
    else:
        config = load_config(Path(args.config))

    if args.device is not None:
        config.run.device = args.device
    if args.reward is not None:
        config.env.reward = args.reward
    if args.start_date is not None:
        config.data.start_date = args.start_date
    if args.end_date is not None:
        config.data.end_date = args.end_date
    if args.trading_period is not None:
        config.env.trading_period = args.trading_period
    if args.initial_capital is not None:
        config.env.initial_capital = args.initial_capital
    if args.transaction_cost_bps is not None:
        config.env.transaction_cost_bps = args.transaction_cost_bps
    if args.slippage_bps is not None:
        config.env.slippage_bps = args.slippage_bps
    if args.action_low is not None:
        config.env.action_low = args.action_low
    if args.action_high is not None:
        config.env.action_high = args.action_high

    eval_cfg = config.eval
    if args.episodes is not None:
        eval_cfg.num_episodes = args.episodes
    if args.epsilon is not None:
        eval_cfg.epsilon = args.epsilon

    mean_return, metrics, _ = evaluate(
        config,
        checkpoint_path=checkpoint_path,
        episodes=eval_cfg.num_episodes,
        epsilon=eval_cfg.epsilon,
        device=config.run.device,
        eval_seed=eval_cfg.seed,
    )
    summary = {
        "num_episodes": int(eval_cfg.num_episodes),
        "mean_reward_return": float(mean_return),
        "median_reward_return": float(metrics.get("median_reward_return", 0.0)),
        "std_reward_return": float(metrics.get("std_reward_return", 0.0)),
        "mean_return_rate": float(metrics.get("mean_return_rate", 0.0)),
        "mean_return_rate_pct": float(metrics.get("mean_return_rate", 0.0) * 100.0),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "initial_state_none_episodes": int(metrics.get("initial_state_none_episodes", 0.0)),
        "zero_step_episodes": int(metrics.get("zero_step_episodes", 0.0)),
        "eval_config": {
            "seed": int(eval_cfg.seed),
            "fixed_windows": bool(eval_cfg.fixed_windows),
            "fixed_windows_seed": eval_cfg.fixed_windows_seed,
            "epsilon": float(eval_cfg.epsilon),
            "save_per_episode": bool(eval_cfg.save_per_episode),
        },
        "window_size": int(config.env.window_size),
        "trading_period": int(config.env.trading_period) if config.env.trading_period is not None else None,
        "action_mode": str(config.env.action_mode),
        "initial_capital": float(config.env.initial_capital),
        "transaction_cost_bps": float(config.env.transaction_cost_bps),
        "slippage_bps": float(config.env.slippage_bps),
        "data_range": {
            "start_date": config.data.start_date,
            "end_date": config.data.end_date,
        },
    }
    output_path = run_dir / "eval_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Eval complete | mean_reward_return %.4f | mean_return_rate %.2f%% | sharpe_ratio %.4f | win_rate %.2f",
        summary["mean_reward_return"],
        summary["mean_return_rate_pct"],
        summary["sharpe_ratio"],
        summary["win_rate"],
    )
    logger.info("Evaluation summary saved to %s", output_path)


if __name__ == "__main__":
    main()
