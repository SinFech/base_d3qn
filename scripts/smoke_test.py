from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.algos.d3qn.trainer import evaluate, load_config, train
from rl.utils.logging import setup_run_logger
from rl.utils.path import build_run_name, build_run_paths


def _run_smoke(config_path: str, run_prefix: str) -> None:
    config = load_config(Path(config_path))
    config.train.num_episodes = 1
    config.train.max_steps_per_episode = 5
    config.train.log_interval = 1
    config.train.checkpoint_interval = 1
    config.env.trading_period = 200
    config.run.seed = 123

    run_name = build_run_name(run_prefix)
    run_paths = build_run_paths(Path("runs"), run_name)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger("smoke_test", run_paths.run_dir)

    train(config, run_paths)

    checkpoint_path = run_paths.checkpoints_dir / "checkpoint_latest.pt"
    mean_return, _, _ = evaluate(
        config,
        checkpoint_path=checkpoint_path,
        episodes=1,
        epsilon=config.train.eval_epsilon,
        device=config.run.device,
    )
    logger.info("%s completed. Mean return: %.2f", run_prefix, mean_return)


def main() -> None:
    _run_smoke("configs/default.yaml", "smoke_test_raw")
    _run_smoke("configs/signature_smoke.yaml", "smoke_test_signature")


if __name__ == "__main__":
    main()
