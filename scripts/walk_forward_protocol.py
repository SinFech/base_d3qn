from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.algos.d3qn.trainer import evaluate as evaluate_d3qn
from rl.algos.d3qn.trainer import load_config as load_d3qn_config
from rl.algos.d3qn.trainer import save_config as save_d3qn_config
from rl.algos.d3qn.trainer import train as train_d3qn
from rl.algos.ppo.trainer import evaluate as evaluate_ppo
from rl.algos.ppo.trainer import load_config as load_ppo_config
from rl.algos.ppo.trainer import save_config as save_ppo_config
from rl.algos.ppo.trainer import train as train_ppo
from rl.utils.path import build_run_paths


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


DEFAULT_FOLDS = [
    FoldSpec(
        fold_id="f1",
        train_start="2014-01-01",
        train_end="2018-12-31",
        test_start="2019-01-01",
        test_end="2022-12-31",
    ),
    FoldSpec(
        fold_id="f2",
        train_start="2014-01-01",
        train_end="2019-12-31",
        test_start="2020-01-01",
        test_end="2023-12-31",
    ),
    FoldSpec(
        fold_id="f3",
        train_start="2014-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2024-02-09",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standardized walk-forward protocol with multiple seeds for PPO/D3QN.",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=["d3qn", "ppo"],
        default=["ppo", "d3qn"],
        help="Algorithms to run.",
    )
    parser.add_argument("--d3qn-config", type=str, default="configs/d3qn_signature_capital.yaml")
    parser.add_argument("--ppo-config", type=str, default="configs/ppo_signature.yaml")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--folds-json",
        type=str,
        default=None,
        help="Optional JSON file containing fold specs. If omitted, default 3-fold expanding protocol is used.",
    )
    parser.add_argument("--output-dir", type=str, default="runs/walk_forward_protocol")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--max-total-steps", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--trading-period", type=int, default=None)
    parser.add_argument(
        "--train-split",
        type=float,
        default=1.0,
        help="Force env.train_split during protocol runs. Default 1.0 to avoid extra 80/20 shrinking.",
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag appended to run names. Defaults to current timestamp.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_seed_list(raw: str) -> list[int]:
    seeds = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    var = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return float(var**0.5)


def _load_fold_specs(path: Path | None) -> list[FoldSpec]:
    if path is None:
        return DEFAULT_FOLDS.copy()

    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("folds", [])
    if not isinstance(payload, list):
        raise ValueError("folds-json must contain a list or {'folds': [...]} object.")

    folds: list[FoldSpec] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError("Each fold entry must be an object.")
        fold_id = str(row.get("fold_id", f"f{idx + 1}"))
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_start=str(row["train_start"]),
                train_end=str(row["train_end"]),
                test_start=str(row["test_start"]),
                test_end=str(row["test_end"]),
            )
        )
    if not folds:
        raise ValueError("No folds found.")
    return folds


def _apply_common_overrides(config: Any, fold: FoldSpec, seed: int, args: argparse.Namespace) -> Any:
    config.data.start_date = fold.train_start
    config.data.end_date = fold.train_end
    config.run.seed = int(seed)
    if hasattr(config, "env") and hasattr(config.env, "train_split"):
        config.env.train_split = float(args.train_split)

    if args.device is not None:
        config.run.device = args.device
    if args.trading_period is not None:
        config.env.trading_period = args.trading_period
    if args.train_episodes is not None:
        config.train.num_episodes = args.train_episodes
    if args.max_total_steps is not None:
        config.train.max_total_steps = args.max_total_steps
    if args.max_steps_per_episode is not None:
        config.train.max_steps_per_episode = args.max_steps_per_episode

    config.eval.num_episodes = args.eval_episodes
    config.eval.epsilon = args.eval_epsilon
    return config


def _resolve_eval_seed(config: Any) -> int:
    eval_cfg = config.eval
    if bool(getattr(eval_cfg, "fixed_windows", False)):
        fixed_windows_seed = getattr(eval_cfg, "fixed_windows_seed", None)
        if fixed_windows_seed is not None:
            return int(fixed_windows_seed)
    return int(eval_cfg.seed)


def _build_eval_summary(config: Any, mean_return: float, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "num_episodes": int(config.eval.num_episodes),
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
            "seed": int(config.eval.seed),
            "fixed_windows": bool(getattr(config.eval, "fixed_windows", False)),
            "fixed_windows_seed": getattr(config.eval, "fixed_windows_seed", None),
            "epsilon": float(config.eval.epsilon),
        },
        "window_size": int(config.env.window_size),
        "trading_period": int(config.env.trading_period) if config.env.trading_period is not None else None,
        "data_range": {
            "start_date": config.data.start_date,
            "end_date": config.data.end_date,
        },
    }


def _evaluate_split(
    algo: str,
    evaluate_fn: Callable[..., tuple[float, dict[str, float], list]],
    base_config: Any,
    checkpoint_path: Path,
    split_start: str,
    split_end: str,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.data.start_date = split_start
    config.data.end_date = split_end

    eval_seed = _resolve_eval_seed(config)
    mean_return, metrics, _ = evaluate_fn(
        config,
        checkpoint_path=checkpoint_path,
        episodes=config.eval.num_episodes,
        epsilon=config.eval.epsilon,
        device=config.run.device,
        eval_seed=eval_seed,
    )
    summary = _build_eval_summary(config, mean_return, metrics)
    summary["algo"] = algo
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _build_result_row(
    algo: str,
    fold: FoldSpec,
    seed: int,
    run_dir: Path,
    is_summary: dict[str, Any],
    oos_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "algo": algo,
        "fold_id": fold.fold_id,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "is_sharpe_ratio": _safe_float(is_summary.get("sharpe_ratio")),
        "oos_sharpe_ratio": _safe_float(oos_summary.get("sharpe_ratio")),
        "is_return_pct": _safe_float(is_summary.get("mean_return_rate_pct")),
        "oos_return_pct": _safe_float(oos_summary.get("mean_return_rate_pct")),
        "is_reward_return": _safe_float(is_summary.get("mean_reward_return")),
        "oos_reward_return": _safe_float(oos_summary.get("mean_reward_return")),
        "is_win_rate": _safe_float(is_summary.get("win_rate")),
        "oos_win_rate": _safe_float(oos_summary.get("win_rate")),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_by_algo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["algo"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for algo, algo_rows in sorted(grouped.items()):
        oos_sharpes = [_safe_float(r["oos_sharpe_ratio"]) for r in algo_rows]
        oos_returns = [_safe_float(r["oos_return_pct"]) for r in algo_rows]
        oos_rewards = [_safe_float(r["oos_reward_return"]) for r in algo_rows]
        is_sharpes = [_safe_float(r["is_sharpe_ratio"]) for r in algo_rows]

        fold_group: dict[str, list[dict[str, Any]]] = {}
        for row in algo_rows:
            fold_group.setdefault(str(row["fold_id"]), []).append(row)
        fold_summary_rows = _summarize_by_algo_fold(algo_rows)
        worst_fold_id = ""
        worst_fold_oos_sharpe_mean = 0.0
        worst_fold_oos_return_pct_mean = 0.0
        worst_fold_oos_reward_mean = 0.0
        if fold_summary_rows:
            worst_fold = min(fold_summary_rows, key=lambda item: _safe_float(item["oos_sharpe_mean"]))
            worst_fold_id = str(worst_fold["fold_id"])
            worst_fold_oos_sharpe_mean = _safe_float(worst_fold["oos_sharpe_mean"])
            worst_fold_oos_return_pct_mean = _safe_float(worst_fold["oos_return_pct_mean"])
            worst_fold_oos_reward_mean = _safe_float(worst_fold["oos_reward_mean"])

        summary_rows.append(
            {
                "algo": algo,
                "runs": len(algo_rows),
                "folds": len(fold_group),
                "oos_sharpe_mean": _mean(oos_sharpes),
                "oos_sharpe_std": _sample_std(oos_sharpes),
                "oos_return_pct_mean": _mean(oos_returns),
                "oos_return_pct_std": _sample_std(oos_returns),
                "oos_reward_mean": _mean(oos_rewards),
                "oos_reward_std": _sample_std(oos_rewards),
                "is_sharpe_mean": _mean(is_sharpes),
                "is_sharpe_std": _sample_std(is_sharpes),
                "worst_fold_id": worst_fold_id,
                "worst_fold_oos_sharpe_mean": worst_fold_oos_sharpe_mean,
                "worst_fold_oos_return_pct_mean": worst_fold_oos_return_pct_mean,
                "worst_fold_oos_reward_mean": worst_fold_oos_reward_mean,
            }
        )
    return summary_rows


def _summarize_by_algo_fold(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["algo"]), str(row["fold_id"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (algo, fold_id), fold_rows in sorted(grouped.items()):
        oos_sharpes = [_safe_float(r["oos_sharpe_ratio"]) for r in fold_rows]
        oos_returns = [_safe_float(r["oos_return_pct"]) for r in fold_rows]
        oos_rewards = [_safe_float(r["oos_reward_return"]) for r in fold_rows]
        is_sharpes = [_safe_float(r["is_sharpe_ratio"]) for r in fold_rows]
        summary_rows.append(
            {
                "algo": algo,
                "fold_id": fold_id,
                "runs": len(fold_rows),
                "oos_sharpe_mean": _mean(oos_sharpes),
                "oos_sharpe_std": _sample_std(oos_sharpes),
                "oos_return_pct_mean": _mean(oos_returns),
                "oos_return_pct_std": _sample_std(oos_returns),
                "oos_reward_mean": _mean(oos_rewards),
                "oos_reward_std": _sample_std(oos_rewards),
                "is_sharpe_mean": _mean(is_sharpes),
                "is_sharpe_std": _sample_std(is_sharpes),
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    if not (0.0 < float(args.train_split) <= 1.0):
        raise ValueError("--train-split must be in the range (0, 1].")
    output_dir = Path(args.output_dir)
    run_tag = (args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")).strip().replace(" ", "_")
    if not run_tag:
        raise ValueError("--run-tag cannot be empty after trimming whitespace.")
    seeds = _parse_seed_list(args.seeds)
    folds = _load_fold_specs(Path(args.folds_json) if args.folds_json else None)

    algo_map: dict[str, dict[str, Any]] = {
        "d3qn": {
            "config_path": Path(args.d3qn_config),
            "load_config": load_d3qn_config,
            "save_config": save_d3qn_config,
            "train": train_d3qn,
            "evaluate": evaluate_d3qn,
        },
        "ppo": {
            "config_path": Path(args.ppo_config),
            "load_config": load_ppo_config,
            "save_config": save_ppo_config,
            "train": train_ppo,
            "evaluate": evaluate_ppo,
        },
    }

    run_plan: list[tuple[str, FoldSpec, int]] = []
    for algo in args.algos:
        for fold in folds:
            for seed in seeds:
                run_plan.append((algo, fold, seed))

    if args.max_runs is not None:
        run_plan = run_plan[: args.max_runs]

    print(
        "Protocol plan:",
        f"algos={args.algos}, folds={len(folds)}, seeds={len(seeds)}, total_runs={len(run_plan)}",
    )
    print(f"Run tag: {run_tag}")
    print(f"Train split override: {args.train_split}")
    if args.dry_run:
        print("Dry-run mode is enabled. No training/evaluation will be executed.")

    result_rows: list[dict[str, Any]] = []
    for idx, (algo, fold, seed) in enumerate(run_plan, start=1):
        run_name = f"{algo}_{fold.fold_id}_s{seed}_{run_tag}"
        run_paths = build_run_paths(output_dir, run_name)
        checkpoint_path = run_paths.checkpoints_dir / "checkpoint_latest.pt"
        is_summary_path = run_paths.run_dir / "eval_is_summary.json"
        oos_summary_path = run_paths.run_dir / "eval_oos_summary.json"

        print(
            f"[{idx}/{len(run_plan)}] algo={algo} fold={fold.fold_id} seed={seed} "
            f"train={fold.train_start}~{fold.train_end} test={fold.test_start}~{fold.test_end}"
        )
        if args.skip_existing and checkpoint_path.exists() and is_summary_path.exists() and oos_summary_path.exists():
            print(f"  skipping existing run: {run_paths.run_dir}")
            is_summary = _load_json(is_summary_path)
            oos_summary = _load_json(oos_summary_path)
            result_rows.append(_build_result_row(algo, fold, seed, run_paths.run_dir, is_summary, oos_summary))
            continue

        if args.dry_run:
            print(f"  planned run_dir={run_paths.run_dir}")
            continue

        impl = algo_map[algo]
        config = impl["load_config"](impl["config_path"])
        config = _apply_common_overrides(config, fold, seed, args)

        run_paths.run_dir.mkdir(parents=True, exist_ok=True)
        impl["save_config"](config, run_paths.config_resolved)
        impl["train"](config, run_paths)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}")

        is_summary = _evaluate_split(
            algo,
            impl["evaluate"],
            config,
            checkpoint_path,
            split_start=fold.train_start,
            split_end=fold.train_end,
        )
        oos_summary = _evaluate_split(
            algo,
            impl["evaluate"],
            config,
            checkpoint_path,
            split_start=fold.test_start,
            split_end=fold.test_end,
        )
        _write_json(is_summary_path, is_summary)
        _write_json(oos_summary_path, oos_summary)

        result_rows.append(_build_result_row(algo, fold, seed, run_paths.run_dir, is_summary, oos_summary))

    if args.dry_run:
        print("Dry-run complete.")
        return

    if not result_rows:
        print("No completed runs were collected.")
        return

    results_csv = output_dir / "results.csv"
    fold_summary_csv = output_dir / "summary_by_algo_fold.csv"
    summary_csv = output_dir / "summary_by_algo.csv"
    _write_csv(results_csv, result_rows)
    fold_summary_rows = _summarize_by_algo_fold(result_rows)
    _write_csv(fold_summary_csv, fold_summary_rows)
    summary_rows = _summarize_by_algo(result_rows)
    _write_csv(summary_csv, summary_rows)

    print(f"Saved detailed results: {results_csv}")
    print(f"Saved algo+fold summary: {fold_summary_csv}")
    print(f"Saved algo summary: {summary_csv}")
    print("Leaderboard (sorted by worst_fold_oos_sharpe_mean, then oos_sharpe_mean):")
    sorted_rows = sorted(
        summary_rows,
        key=lambda x: (
            _safe_float(x["worst_fold_oos_sharpe_mean"]),
            _safe_float(x["oos_sharpe_mean"]),
        ),
        reverse=True,
    )
    for row in sorted_rows:
        print(
            f"  {row['algo']}: runs={row['runs']}, "
            f"decision_metric_worst_fold_oos_sharpe_mean={row['worst_fold_oos_sharpe_mean']:.4f}, "
            f"oos_sharpe_mean={row['oos_sharpe_mean']:.4f}, "
            f"oos_sharpe_std={row['oos_sharpe_std']:.4f}, "
            f"worst_fold={row['worst_fold_id']}, "
            f"oos_return_pct_mean={row['oos_return_pct_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
