from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.utils.logging import setup_run_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep runner")
    parser.add_argument("--sweep-config", type=str, default="configs/sweep.yaml")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--gpu-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sweep_config(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    return data or {}


def build_override_grid(overrides: Dict[str, List]) -> List[Dict[str, object]]:
    if not overrides:
        return [{}]
    keys = list(overrides.keys())
    values = [overrides[key] for key in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def build_command(
    base_config: str,
    output_dir: str,
    run_name: str,
    overrides: Dict[str, object],
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        base_config,
        "--output-dir",
        output_dir,
        "--run-name",
        run_name,
    ]
    for key, value in overrides.items():
        cmd.extend(["--override", f"{key}={value}"])
    return cmd


def run_sweep() -> None:
    args = parse_args()
    sweep_cfg = load_sweep_config(Path(args.sweep_config))

    base_config = sweep_cfg.get("base_config", "configs/default.yaml")
    output_dir = args.output_dir or sweep_cfg.get("output_dir", "runs/sweeps")
    run_prefix = args.run_prefix or sweep_cfg.get("run_prefix", "d3qn_sweep")
    max_parallel = args.max_parallel or sweep_cfg.get("max_parallel", 1)
    gpu_id = args.gpu_id or str(sweep_cfg.get("gpu_id", 0))
    logger = setup_run_logger("sweep", Path(output_dir))

    overrides = sweep_cfg.get("overrides", {})
    grid = build_override_grid(overrides)
    if args.max_runs is not None:
        grid = grid[: args.max_runs]

    if not grid:
        logger.warning("No sweep runs configured.")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    running = []
    completed = 0
    total = len(grid)

    for idx, override_set in enumerate(grid):
        run_name = f"{run_prefix}_{idx:03d}"
        cmd = build_command(base_config, output_dir, run_name, override_set)
        if args.dry_run:
            logger.info("DRY RUN: %s", " ".join(cmd))
            continue

        while len(running) >= max_parallel:
            for proc in running[:]:
                if proc.poll() is not None:
                    running.remove(proc)
                    completed += 1
            time.sleep(0.2)

        logger.info("Starting %s/%s: %s", completed + len(running) + 1, total, run_name)
        proc = subprocess.Popen(cmd, env=env)
        running.append(proc)

    while running:
        for proc in running[:]:
            if proc.poll() is not None:
                running.remove(proc)
                completed += 1
        time.sleep(0.2)

    logger.info("Sweep finished: %s/%s runs completed", completed, total)


if __name__ == "__main__":
    run_sweep()
