from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.utils.logging import setup_run_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sweep runs")
    parser.add_argument("--runs-dir", type=str, default="runs/sweeps")
    parser.add_argument("--last-n", type=int, default=10)
    parser.add_argument("--sort-by", type=str, default="mean_last_n")
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def summarize_run(run_dir: Path, last_n: int) -> Dict[str, object]:
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config_resolved.yaml"

    if not metrics_path.exists():
        return {}

    df = pd.read_csv(metrics_path)
    if df.empty:
        return {}

    config = load_config(config_path)

    episode_return = df["episode_return"]
    final_return = float(episode_return.iloc[-1])
    mean_last_n = float(episode_return.tail(last_n).mean())
    best_return = float(episode_return.max())

    summary = {
        "run_name": run_dir.name,
        "final_return": final_return,
        "mean_last_n": mean_last_n,
        "best_return": best_return,
    }

    env_cfg = config.get("env", {})
    agent_cfg = config.get("agent", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})

    summary["reward"] = env_cfg.get("reward")
    summary["learning_rate"] = agent_cfg.get("learning_rate")
    summary["eps_steps"] = agent_cfg.get("eps_steps")
    summary["num_episodes"] = train_cfg.get("num_episodes")
    summary["trading_period"] = env_cfg.get("trading_period")
    summary["train_split"] = env_cfg.get("train_split")
    summary["data_start_date"] = data_cfg.get("start_date")
    summary["data_end_date"] = data_cfg.get("end_date")

    return summary


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    logger = setup_run_logger("sweep_report", runs_dir)

    rows: List[Dict[str, object]] = []
    for run_dir in sorted(runs_dir.iterdir() if runs_dir.exists() else []):
        if not run_dir.is_dir():
            continue
        summary = summarize_run(run_dir, args.last_n)
        if summary:
            rows.append(summary)

    if not rows:
        logger.info("No runs found.")
        return

    df = pd.DataFrame(rows)
    if args.sort_by in df.columns:
        df = df.sort_values(by=args.sort_by, ascending=args.ascending)

    key_fields = [
        "reward",
        "learning_rate",
        "eps_steps",
        "trading_period",
        "train_split",
        "data_start_date",
        "data_end_date",
    ]
    seen: Dict[tuple, List[str]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if any(value is not None for value in key):
            seen.setdefault(key, []).append(str(row.get("run_name")))

    duplicates = {key: names for key, names in seen.items() if len(names) > 1}
    if duplicates:
        logger.warning("Duplicate sweep configs detected.")
        for key, names in duplicates.items():
            details = ", ".join(f"{field}={value}" for field, value in zip(key_fields, key))
            logger.warning("- %s | %s", ", ".join(names), details)

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)

    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
