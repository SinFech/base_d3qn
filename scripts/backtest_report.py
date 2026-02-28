from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "matplotlib-codex-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.algos.d3qn import trainer as d3qn_trainer
from rl.algos.ppo import trainer as ppo_trainer
from rl.algos.sac import trainer as sac_trainer
from rl.envs.make_env import filter_date_range, load_price_data, make_env
from rl.utils.checkpoint import load_checkpoint


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a professional single-run backtest tear sheet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--algo", type=str, default="auto", choices=["auto", "d3qn", "ppo", "sac"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--full-range", action="store_true")
    parser.add_argument("--reward", type=str, default=None, choices=["profit", "sr", "sr_enhanced"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--trading-period", type=int, default=None)
    parser.add_argument("--action-mode", type=str, default=None, choices=["discrete", "discrete_capital", "continuous"])
    parser.add_argument("--sell-mode", type=str, default=None, choices=["all", "one", "all_cap", "one_plus"])
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-exposure-ratio", type=float, default=None)
    parser.add_argument("--initial-capital", type=float, default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--invalid-sell-penalty", type=float, default=None)
    parser.add_argument("--action-low", type=float, default=None)
    parser.add_argument("--action-high", type=float, default=None)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--rolling-window", type=int, default=30)
    return parser.parse_args()


def _resolve_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _detect_algo(checkpoint: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    policy_state = checkpoint.get("policy_state", {})
    target_state = checkpoint.get("target_state", {})
    if isinstance(policy_state, dict) and "actor" in policy_state:
        return "sac"
    if isinstance(target_state, dict) and len(target_state) == 0:
        return "ppo"
    return "d3qn"


def _default_config_path(algo: str) -> Path:
    if algo == "ppo":
        return Path("configs/ppo_signature.yaml")
    if algo == "sac":
        return Path("configs/sac_signature.yaml")
    return Path("configs/default.yaml")


def _load_runtime_config(algo: str, checkpoint: dict[str, Any], config_path: Optional[str]):
    if "config" in checkpoint:
        raw_cfg = checkpoint["config"]
        if algo == "ppo":
            return ppo_trainer.config_from_dict(raw_cfg)
        if algo == "sac":
            return sac_trainer.config_from_dict(raw_cfg)
        return d3qn_trainer.config_from_dict(raw_cfg)
    config_file = Path(config_path) if config_path else _default_config_path(algo)
    if algo == "ppo":
        return ppo_trainer.load_config(config_file)
    if algo == "sac":
        return sac_trainer.load_config(config_file)
    return d3qn_trainer.load_config(config_file)


def _set_attr_if_present(target: Any, name: str, value: Any) -> None:
    if hasattr(target, name):
        setattr(target, name, value)


def _apply_overrides(config: Any, args: argparse.Namespace) -> None:
    if args.device is not None:
        _set_attr_if_present(config.run, "device", args.device)

    if args.reward is not None:
        _set_attr_if_present(config.env, "reward", args.reward)
    if args.start_date is not None:
        _set_attr_if_present(config.data, "start_date", args.start_date)
    if args.end_date is not None:
        _set_attr_if_present(config.data, "end_date", args.end_date)
    if args.action_mode is not None:
        _set_attr_if_present(config.env, "action_mode", args.action_mode)
    if args.sell_mode is not None:
        _set_attr_if_present(config.env, "sell_mode", args.sell_mode)
    if args.full_range:
        _set_attr_if_present(config.env, "trading_period", None)
    elif args.trading_period is not None:
        _set_attr_if_present(config.env, "trading_period", args.trading_period)
    if args.max_positions is not None and hasattr(config.env, "max_positions"):
        config.env.max_positions = None if args.max_positions < 1 else int(args.max_positions)
    if args.max_exposure_ratio is not None:
        _set_attr_if_present(config.env, "max_exposure_ratio", float(args.max_exposure_ratio))
    if args.initial_capital is not None:
        _set_attr_if_present(config.env, "initial_capital", float(args.initial_capital))
    if args.transaction_cost_bps is not None:
        _set_attr_if_present(config.env, "transaction_cost_bps", float(args.transaction_cost_bps))
    if args.slippage_bps is not None:
        _set_attr_if_present(config.env, "slippage_bps", float(args.slippage_bps))
    if args.invalid_sell_penalty is not None:
        _set_attr_if_present(config.env, "invalid_sell_penalty", float(args.invalid_sell_penalty))
    if args.action_low is not None:
        _set_attr_if_present(config.env, "action_low", float(args.action_low))
    if args.action_high is not None:
        _set_attr_if_present(config.env, "action_high", float(args.action_high))


def _prepare_data(config: Any) -> pd.DataFrame:
    df = load_price_data(
        Path(config.data.path),
        price_column=config.data.price_column,
        close_column=config.data.close_column,
        date_column=config.data.date_column,
    )
    return filter_date_range(
        df,
        date_column=config.data.date_column,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
    )


def _compute_start_range(data_len: int, window_size: int, trading_period: Optional[int]) -> tuple[int, int]:
    min_start = int(window_size) - 1
    max_start = data_len - 1 if trading_period is None else data_len - int(trading_period)
    return min_start, max_start


def _unwrap_base_env(env: Any) -> Any:
    current = env
    for _ in range(20):
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    return current


def _extract_position_units(base_env: Any) -> float:
    if hasattr(base_env, "position_units"):
        return float(base_env.position_units)
    if hasattr(base_env, "_position_units"):
        return float(base_env._position_units())
    if hasattr(base_env, "agent_positions"):
        positions = list(base_env.agent_positions)
        if not positions:
            return 0.0
        if isinstance(positions[0], tuple):
            return float(sum(float(qty) for _, qty in positions))
        return float(len(positions))
    if hasattr(base_env, "position"):
        return float(base_env.position)
    return 0.0


def _extract_price(base_env: Any, df: pd.DataFrame, index: int) -> float:
    if hasattr(base_env, "_current_price"):
        return float(base_env._current_price())
    clipped = min(max(int(index), 0), len(df) - 1)
    return float(df.iloc[clipped]["Close"])


def _extract_equity(base_env: Any, price: float) -> float:
    if hasattr(base_env, "_equity"):
        return float(base_env._equity(price))
    if hasattr(base_env, "equity_end"):
        return float(base_env.equity_end)
    if (
        hasattr(base_env, "init_price")
        and hasattr(base_env, "realized_pnl")
        and hasattr(base_env, "agent_open_position_value")
    ):
        return float(base_env.init_price + base_env.realized_pnl + base_env.agent_open_position_value)
    if hasattr(base_env, "cash"):
        cash = float(base_env.cash)
        units = _extract_position_units(base_env)
        return cash + units * float(price)
    return float("nan")


def _extract_position_value(base_env: Any, price: float) -> float:
    if hasattr(base_env, "_position_value"):
        return float(base_env._position_value(price))
    units = _extract_position_units(base_env)
    return units * float(price)


def _extract_cash(base_env: Any, equity: float, position_value: float) -> float:
    if hasattr(base_env, "cash"):
        return float(base_env.cash)
    if np.isfinite(equity) and np.isfinite(position_value):
        return float(equity - position_value)
    return float("nan")


def _continuous_action_label(value: float, tol: float = 1e-6) -> str:
    if value > tol:
        return "long"
    if value < -tol:
        return "short"
    return "flat"


def _build_action_name_map(
    action_number: int,
    sell_mode: str,
    buy_fractions: Optional[list[float]],
    sell_fractions: Optional[list[float]],
) -> dict[int, str]:
    buys = [float(v) for v in (buy_fractions or [])]
    sells = [float(v) for v in (sell_fractions or [])]
    if buys and not sells:
        sells = [1.0]
    if sells and not buys:
        buys = [1.0]
    if buys or sells:
        action_map: dict[int, str] = {0: "hold"}
        offset = 1
        for fraction in buys:
            action_map[offset] = f"buy_{int(round(fraction * 100.0))}pct"
            offset += 1
        for fraction in sells:
            action_map[offset] = "sell_all" if abs(fraction - 1.0) < 1e-12 else f"sell_{int(round(fraction * 100.0))}pct"
            offset += 1
        return action_map
    if action_number >= 4:
        if sell_mode == "one_plus":
            return {0: "hold", 1: "buy", 2: "sell_one", 3: "sell_all"}
        return {0: "hold", 1: "buy", 2: "sell", 3: "sell_all"}
    return {0: "hold", 1: "buy", 2: "sell"}


def _safe_std(values: pd.Series) -> float:
    if values.size < 2:
        return 0.0
    std = float(values.std(ddof=1))
    if not np.isfinite(std):
        return 0.0
    return std


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    max_duration = 0
    current = 0
    for value in drawdown.to_numpy(dtype=float):
        if value < 0:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return int(max_duration)


def _annualized_return(total_return: float, periods: int, periods_per_year: float) -> float:
    if periods <= 0:
        return 0.0
    gross = 1.0 + float(total_return)
    if gross <= 0:
        return -1.0
    return float(gross ** (periods_per_year / periods) - 1.0)


def _safe_div(num: float, den: float) -> float:
    if abs(den) <= EPS or not np.isfinite(den):
        return 0.0
    value = num / den
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        if math.isfinite(v):
            return v
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def _plot_equity(ts: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(ts.index, ts["equity"], label="Strategy Equity", linewidth=2.0)
    axes[0].plot(ts.index, ts["benchmark_equity"], label="Buy & Hold Benchmark", linewidth=1.8, alpha=0.9)
    axes[0].set_ylabel("Equity")
    axes[0].set_title("Equity Curve vs Buy & Hold")
    axes[0].legend()

    axes[1].plot(ts.index, ts["cumulative_return"] * 100.0, label="Strategy", linewidth=2.0)
    axes[1].plot(ts.index, ts["benchmark_cumulative_return"] * 100.0, label="Benchmark", linewidth=1.8, alpha=0.9)
    axes[1].set_ylabel("Cumulative Return (%)")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_drawdown(ts: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.fill_between(ts.index, ts["drawdown"] * 100.0, 0.0, alpha=0.35, label="Strategy Drawdown")
    ax.plot(ts.index, ts["benchmark_drawdown"] * 100.0, linewidth=1.5, alpha=0.9, label="Benchmark Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_title("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_rolling_sharpe(ts: pd.DataFrame, rolling_window: int, periods_per_year: float, output_path: Path) -> None:
    rolling_mean = ts["strategy_return"].rolling(rolling_window, min_periods=max(2, rolling_window // 3)).mean()
    rolling_std = ts["strategy_return"].rolling(rolling_window, min_periods=max(2, rolling_window // 3)).std(ddof=1)
    rolling_sharpe = np.sqrt(periods_per_year) * rolling_mean / (rolling_std + EPS)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(ts.index, rolling_sharpe, linewidth=1.8, label=f"Rolling Sharpe ({rolling_window})")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Date")
    ax.set_title("Rolling Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_return_distribution(ts: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(ts["strategy_return"], bins=40, alpha=0.65, label="Strategy", density=True)
    ax.hist(ts["benchmark_return"], bins=40, alpha=0.45, label="Benchmark", density=True)
    ax.axvline(float(ts["strategy_return"].mean()), color="tab:blue", linestyle="--", linewidth=1.2)
    ax.axvline(float(ts["benchmark_return"].mean()), color="tab:orange", linestyle="--", linewidth=1.2)
    ax.set_title("Step Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_monthly_heatmap(ts: pd.DataFrame, output_path: Path) -> None:
    monthly = ts["equity"].resample("ME").last().pct_change().dropna()
    if monthly.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "Not enough data for monthly heatmap", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    years = sorted(set(int(ts_idx.year) for ts_idx in monthly.index))
    heat = np.full((len(years), 12), np.nan, dtype=float)
    for dt, value in monthly.items():
        year_idx = years.index(int(dt.year))
        month_idx = int(dt.month) - 1
        heat[year_idx, month_idx] = float(value) * 100.0

    max_abs = float(np.nanmax(np.abs(heat))) if np.isfinite(np.nanmax(np.abs(heat))) else 1.0
    max_abs = max(max_abs, 1.0)

    fig, ax = plt.subplots(figsize=(14, max(3.0, 0.6 * len(years) + 2.0)))
    image = ax.imshow(heat, cmap="RdYlGn", aspect="auto", vmin=-max_abs, vmax=max_abs)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels([str(y) for y in years])
    ax.set_title("Monthly Returns Heatmap (%)")
    fig.colorbar(image, ax=ax, fraction=0.028, pad=0.015)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_position_and_action(ts: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    axes[0].plot(ts.index, ts["position_units"], linewidth=1.8, label="Position Units")
    axes[0].set_ylabel("Units")
    axes[0].set_title("Position and Action Trace")
    axes[0].legend()

    axes[1].plot(ts.index, ts["action_value"], linewidth=1.2, label="Action Value")
    axes[1].set_ylabel("Action")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_markdown_report(summary: dict[str, Any], figures_rel: dict[str, str]) -> str:
    actions = summary["action_distribution"]
    metric_rows = [
        ("Total Return (%)", summary["performance"]["total_return_pct"], summary["benchmark"]["total_return_pct"]),
        ("Annualized Return (%)", summary["performance"]["annualized_return_pct"], summary["benchmark"]["annualized_return_pct"]),
        ("Annualized Volatility (%)", summary["performance"]["annualized_volatility_pct"], summary["benchmark"]["annualized_volatility_pct"]),
        ("Sharpe", summary["performance"]["sharpe"], summary["benchmark"]["sharpe"]),
        ("Sortino", summary["performance"]["sortino"], summary["benchmark"]["sortino"]),
        ("Calmar", summary["performance"]["calmar"], summary["benchmark"]["calmar"]),
        ("Max Drawdown (%)", summary["performance"]["max_drawdown_pct"], summary["benchmark"]["max_drawdown_pct"]),
        ("Win Rate (%)", summary["performance"]["win_rate_pct"], summary["benchmark"]["win_rate_pct"]),
    ]
    action_lines = "\n".join(
        f"- `{name}`: {count} ({rate * 100.0:.2f}%)"
        for name, (count, rate) in sorted(actions.items(), key=lambda item: item[0])
    )

    metrics_table = "\n".join(
        f"| {name} | {strat:.4f} | {bench:.4f} |" for name, strat, bench in metric_rows
    )

    return (
        "# Backtest Tear Sheet\n\n"
        "## Run Overview\n\n"
        f"- Checkpoint: `{summary['run']['checkpoint']}`\n"
        f"- Algorithm: `{summary['run']['algo']}`\n"
        f"- Date range: `{summary['run']['start_timestamp']}` to `{summary['run']['end_timestamp']}`\n"
        f"- Steps: `{summary['run']['num_steps']}`\n"
        f"- Trading period: `{summary['run']['trading_period']}`\n\n"
        "## Performance Comparison\n\n"
        "| Metric | Strategy | Benchmark |\n"
        "|---|---:|---:|\n"
        f"{metrics_table}\n\n"
        "## Risk and Trading Diagnostics\n\n"
        f"- Max drawdown duration (steps): `{summary['performance']['max_drawdown_duration_steps']}`\n"
        f"- Information ratio: `{summary['performance']['information_ratio']:.4f}`\n"
        f"- Alpha (annualized): `{summary['performance']['alpha_annualized']:.4f}`\n"
        f"- Beta: `{summary['performance']['beta']:.4f}`\n"
        f"- Exposure mean/max: `{summary['performance']['exposure_mean_pct']:.2f}% / {summary['performance']['exposure_max_pct']:.2f}%`\n"
        f"- Turnover (units abs): `{summary['performance']['turnover_units_abs']:.4f}`\n"
        f"- Trade count: `{summary['performance']['trade_count']}`\n"
        f"- Reward return (sum): `{summary['performance']['reward_return']:.6f}`\n\n"
        "## Action Distribution\n\n"
        f"{action_lines}\n\n"
        "## Figures\n\n"
        f"![Equity Curve]({figures_rel['equity_vs_benchmark']})\n\n"
        f"![Drawdown]({figures_rel['drawdown']})\n\n"
        f"![Rolling Sharpe]({figures_rel['rolling_sharpe']})\n\n"
        f"![Monthly Returns Heatmap]({figures_rel['monthly_heatmap']})\n\n"
        f"![Return Distribution]({figures_rel['return_distribution']})\n\n"
        f"![Position and Action]({figures_rel['position_action']})\n"
    )


def _build_html_report(summary: dict[str, Any], figures_rel: dict[str, str]) -> str:
    action_items = "".join(
        f"<li><code>{name}</code>: {count} ({rate * 100.0:.2f}%)</li>"
        for name, (count, rate) in sorted(summary["action_distribution"].items(), key=lambda item: item[0])
    )
    metric_rows = [
        ("Total Return (%)", "total_return_pct"),
        ("Annualized Return (%)", "annualized_return_pct"),
        ("Annualized Volatility (%)", "annualized_volatility_pct"),
        ("Sharpe", "sharpe"),
        ("Sortino", "sortino"),
        ("Calmar", "calmar"),
        ("Max Drawdown (%)", "max_drawdown_pct"),
        ("Win Rate (%)", "win_rate_pct"),
    ]
    table_rows = "".join(
        "<tr>"
        f"<td>{name}</td>"
        f"<td>{summary['performance'][key]:.4f}</td>"
        f"<td>{summary['benchmark'][key]:.4f}</td>"
        "</tr>"
        for name, key in metric_rows
    )
    image_blocks = "".join(
        f"<h3>{title}</h3><img src='{path}' alt='{title}'/>"
        for title, path in [
            ("Equity Curve", figures_rel["equity_vs_benchmark"]),
            ("Drawdown", figures_rel["drawdown"]),
            ("Rolling Sharpe", figures_rel["rolling_sharpe"]),
            ("Monthly Returns Heatmap", figures_rel["monthly_heatmap"]),
            ("Return Distribution", figures_rel["return_distribution"]),
            ("Position and Action", figures_rel["position_action"]),
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Backtest Tear Sheet</title>
  <style>
    body {{
      margin: 24px auto;
      max-width: 1080px;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: #111827;
      line-height: 1.45;
      padding: 0 16px;
      background: #f8fafc;
    }}
    .card {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 18px 20px;
      margin-bottom: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      margin: 8px 0 20px 0;
      background: white;
    }}
    code {{
      background: #f3f4f6;
      padding: 1px 5px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <h1>Backtest Tear Sheet</h1>
  <div class="card">
    <h2>Run Overview</h2>
    <ul>
      <li>Checkpoint: <code>{summary['run']['checkpoint']}</code></li>
      <li>Algorithm: <code>{summary['run']['algo']}</code></li>
      <li>Date range: <code>{summary['run']['start_timestamp']}</code> to <code>{summary['run']['end_timestamp']}</code></li>
      <li>Steps: <code>{summary['run']['num_steps']}</code></li>
      <li>Trading period: <code>{summary['run']['trading_period']}</code></li>
    </ul>
  </div>
  <div class="card">
    <h2>Performance Comparison</h2>
    <table>
      <thead>
        <tr><th>Metric</th><th>Strategy</th><th>Benchmark</th></tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>
  <div class="card">
    <h2>Risk and Trading Diagnostics</h2>
    <ul>
      <li>Max drawdown duration (steps): <code>{summary['performance']['max_drawdown_duration_steps']}</code></li>
      <li>Information ratio: <code>{summary['performance']['information_ratio']:.4f}</code></li>
      <li>Alpha (annualized): <code>{summary['performance']['alpha_annualized']:.4f}</code></li>
      <li>Beta: <code>{summary['performance']['beta']:.4f}</code></li>
      <li>Exposure mean/max: <code>{summary['performance']['exposure_mean_pct']:.2f}% / {summary['performance']['exposure_max_pct']:.2f}%</code></li>
      <li>Turnover (units abs): <code>{summary['performance']['turnover_units_abs']:.4f}</code></li>
      <li>Trade count: <code>{summary['performance']['trade_count']}</code></li>
      <li>Reward return (sum): <code>{summary['performance']['reward_return']:.6f}</code></li>
    </ul>
  </div>
  <div class="card">
    <h2>Action Distribution</h2>
    <ul>{action_items}</ul>
  </div>
  <div class="card">
    <h2>Figures</h2>
    {image_blocks}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    run_dir = Path(args.output_dir) if args.output_dir else (_resolve_run_dir(checkpoint_path) / "tear_sheet")
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path, device=args.device or "auto")
    algo = _detect_algo(checkpoint, args.algo)
    config = _load_runtime_config(algo, checkpoint, args.config)
    _apply_overrides(config, args)

    seed = int(args.seed if args.seed is not None else getattr(config.eval, "seed", getattr(config.run, "seed", 42)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    resolved_device = _resolve_device(getattr(config.run, "device", "auto"))
    df = _prepare_data(config)
    if len(df) < int(config.env.window_size):
        raise ValueError(
            f"Not enough rows after date filtering: len(df)={len(df)} < window_size={config.env.window_size}"
        )

    min_start, max_start = _compute_start_range(len(df), int(config.env.window_size), getattr(config.env, "trading_period", None))
    if max_start < min_start:
        raise ValueError(
            "Invalid start_index range "
            f"[{min_start}, {max_start}] for data length {len(df)} and trading_period {getattr(config.env, 'trading_period', None)}"
        )
    start_index = int(args.start_index) if args.start_index is not None else int(min_start)
    if not (min_start <= start_index <= max_start):
        raise ValueError(f"start_index must be in [{min_start}, {max_start}], got {start_index}")

    env = make_env(
        df,
        config.env.reward,
        config.env.window_size,
        resolved_device,
        trading_period=getattr(config.env, "trading_period", None),
        max_positions=getattr(config.env, "max_positions", None),
        max_exposure_ratio=getattr(config.env, "max_exposure_ratio", 1.0),
        sell_mode=getattr(config.env, "sell_mode", "all"),
        buy_fractions=getattr(config.env, "buy_fractions", None),
        sell_fractions=getattr(config.env, "sell_fractions", None),
        action_number=getattr(getattr(config, "agent", None), "action_number", None),
        action_mode=getattr(config.env, "action_mode", "discrete"),
        initial_capital=getattr(config.env, "initial_capital", 100_000.0),
        transaction_cost_bps=getattr(config.env, "transaction_cost_bps", 10.0),
        slippage_bps=getattr(config.env, "slippage_bps", 2.0),
        invalid_sell_penalty=getattr(config.env, "invalid_sell_penalty", 0.1),
        blocked_trade_penalty=getattr(config.env, "blocked_trade_penalty", 0.0),
        min_hold_steps=getattr(config.env, "min_hold_steps", 0),
        trade_cooldown_steps=getattr(config.env, "trade_cooldown_steps", 0),
        dynamic_exposure_enabled=getattr(config.env, "dynamic_exposure_enabled", False),
        dynamic_exposure_vol_window=getattr(config.env, "dynamic_exposure_vol_window", 30),
        dynamic_exposure_min_scale=getattr(config.env, "dynamic_exposure_min_scale", 0.5),
        dynamic_exposure_strength=getattr(config.env, "dynamic_exposure_strength", 1.0),
        allow_short=getattr(config.env, "allow_short", False),
        max_leverage=getattr(config.env, "max_leverage", 1.0),
        action_low=getattr(config.env, "action_low", 0.0),
        action_high=getattr(config.env, "action_high", 1.0),
        min_equity_ratio=getattr(config.env, "min_equity_ratio", 0.2),
        stop_on_bankruptcy=getattr(config.env, "stop_on_bankruptcy", True),
        obs_config=getattr(config.env, "obs", None),
    )
    obs_dim = int(getattr(env, "obs_dim", config.env.window_size))
    base_env = _unwrap_base_env(env)

    action_name_map: dict[int, str] = {}
    d3qn_agent = None
    ppo_model = None
    ppo_action_low = None
    ppo_action_high = None
    sac_actor = None

    if algo == "d3qn":
        config.agent.input_dim = obs_dim
        d3qn_agent = d3qn_trainer.build_agent(config, resolved_device, input_dim=obs_dim)
        d3qn_agent.policy_net.load_state_dict(checkpoint["policy_state"])
        d3qn_agent.target_net.load_state_dict(checkpoint["target_state"])
        d3qn_agent.policy_net.eval()
        d3qn_agent.target_net.eval()
        d3qn_agent.reset_episode()
        action_name_map = _build_action_name_map(
            int(config.agent.action_number),
            getattr(config.env, "sell_mode", "all"),
            getattr(config.env, "buy_fractions", None),
            getattr(config.env, "sell_fractions", None),
        )
    elif algo == "ppo":
        ppo_model = ppo_trainer._build_model(config, obs_dim, resolved_device)
        ppo_model.load_state_dict(checkpoint["policy_state"])
        ppo_model.eval()
        ppo_action_low, ppo_action_high = ppo_trainer._action_bounds(config, resolved_device)
    else:
        sac_actor, _, _, _, _ = sac_trainer._build_networks(config, obs_dim, resolved_device)
        payload = checkpoint.get("policy_state", {})
        if isinstance(payload, dict) and "actor" in payload:
            sac_actor.load_state_dict(payload["actor"])
        else:
            sac_actor.load_state_dict(payload)
        sac_actor.eval()

    env.reset(seed=seed, start_index=start_index)
    state = env.get_state()
    if state is None:
        raise ValueError("Environment returned None state immediately. Check date range / trading_period / window_size.")

    records: list[dict[str, Any]] = []
    action_counter: Counter[str] = Counter()
    reward_return = 0.0
    previous_equity = None
    previous_benchmark_equity = None

    initial_index = int(getattr(base_env, "t", start_index))
    initial_price = _extract_price(base_env, df, initial_index)
    initial_equity = _extract_equity(base_env, initial_price)
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        initial_equity = float(getattr(base_env, "equity_start", getattr(config.env, "initial_capital", 100_000.0)))
    initial_equity = max(float(initial_equity), EPS)

    with torch.no_grad():
        while state is not None:
            row_index = int(getattr(base_env, "t", start_index + len(records)))
            if row_index < 0 or row_index >= len(df):
                break
            timestamp = pd.Timestamp(df.iloc[row_index]["Date"])
            price = _extract_price(base_env, df, row_index)

            if algo == "d3qn":
                action_tensor = d3qn_agent.select_action(state, training=False, epsilon_override=float(args.epsilon))
                action_env = action_tensor
                action_value = float(int(action_tensor.item()))
                action_label = action_name_map.get(int(action_tensor.item()), f"action_{int(action_tensor.item())}")
            elif algo == "ppo":
                state_t = ppo_trainer._to_obs_tensor(state, resolved_device)
                if state_t is None:
                    break
                if rng.random() < float(args.epsilon):
                    action_value = float(rng.uniform(config.env.action_low, config.env.action_high))
                else:
                    dist, _ = ppo_model.distribution(state_t)
                    if args.stochastic_policy:
                        raw_action = dist.sample()
                    else:
                        raw_action = dist.loc if config.ppo.deterministic_eval else dist.sample()
                    squashed = torch.tanh(raw_action)
                    scaled = ppo_trainer._scale_action(squashed, ppo_action_low, ppo_action_high)
                    action_value = float(ppo_trainer._env_action_from_tensor(scaled.squeeze(0)))
                action_env = action_value
                action_label = _continuous_action_label(action_value)
            else:
                state_t = sac_trainer._to_obs_tensor(state, resolved_device)
                if state_t is None:
                    break
                if rng.random() < float(args.epsilon):
                    action_value = float(rng.uniform(config.env.action_low, config.env.action_high))
                else:
                    if args.stochastic_policy:
                        _, action_value = sac_trainer._sample_env_action_from_actor(sac_actor, state_t)
                    else:
                        if config.sac.deterministic_eval:
                            _, action_value = sac_trainer._deterministic_env_action_from_actor(sac_actor, state_t)
                        else:
                            _, action_value = sac_trainer._sample_env_action_from_actor(sac_actor, state_t)
                action_env = action_value
                action_label = _continuous_action_label(action_value)

            reward, done, _ = env.step(action_env)
            reward_value = float(reward.item()) if isinstance(reward, torch.Tensor) else float(reward)
            reward_return += reward_value

            position_units = _extract_position_units(base_env)
            position_value = _extract_position_value(base_env, price)
            equity = _extract_equity(base_env, price)
            if not np.isfinite(equity):
                equity = initial_equity
            cash = _extract_cash(base_env, equity, position_value)
            exposure_ratio = abs(position_value) / max(abs(equity), EPS)

            benchmark_equity = initial_equity * (price / max(initial_price, EPS))
            strategy_return = 0.0 if previous_equity is None else (equity - previous_equity) / max(abs(previous_equity), EPS)
            benchmark_return = (
                0.0
                if previous_benchmark_equity is None
                else (benchmark_equity - previous_benchmark_equity) / max(abs(previous_benchmark_equity), EPS)
            )
            cumulative_return = (equity / initial_equity) - 1.0
            benchmark_cum_return = (benchmark_equity / initial_equity) - 1.0

            records.append(
                {
                    "step": int(len(records)),
                    "index": int(row_index),
                    "timestamp": timestamp,
                    "price": float(price),
                    "action_label": str(action_label),
                    "action_value": float(action_value),
                    "reward": float(reward_value),
                    "equity": float(equity),
                    "cash": float(cash),
                    "position_units": float(position_units),
                    "position_value": float(position_value),
                    "exposure_ratio": float(exposure_ratio),
                    "strategy_return": float(strategy_return),
                    "benchmark_return": float(benchmark_return),
                    "cumulative_return": float(cumulative_return),
                    "benchmark_equity": float(benchmark_equity),
                    "benchmark_cumulative_return": float(benchmark_cum_return),
                }
            )
            action_counter[str(action_label)] += 1

            previous_equity = float(equity)
            previous_benchmark_equity = float(benchmark_equity)
            state = env.get_state()
            if done:
                break

    if not records:
        raise ValueError("Backtest produced zero steps. Try a smaller window or a different start_index.")

    ts = pd.DataFrame.from_records(records)
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])
    ts = ts.set_index("timestamp").sort_index()

    equity_running_max = ts["equity"].cummax()
    benchmark_running_max = ts["benchmark_equity"].cummax()
    ts["drawdown"] = ts["equity"] / equity_running_max - 1.0
    ts["benchmark_drawdown"] = ts["benchmark_equity"] / benchmark_running_max - 1.0

    trades = ts.copy()
    trades["units_delta"] = trades["position_units"].diff().fillna(0.0)
    trades = trades.loc[np.abs(trades["units_delta"]) > EPS, ["index", "price", "action_label", "action_value", "position_units", "units_delta", "equity", "cash"]]

    periods = int(len(ts))
    periods_per_year = float(args.periods_per_year)
    rf_annual = float(args.risk_free_rate)
    rf_step = (1.0 + rf_annual) ** (1.0 / max(periods_per_year, 1.0)) - 1.0

    strategy_returns = ts["strategy_return"]
    benchmark_returns = ts["benchmark_return"]
    excess_returns = strategy_returns - benchmark_returns

    total_return = float(ts["cumulative_return"].iloc[-1])
    bench_total_return = float(ts["benchmark_cumulative_return"].iloc[-1])
    ann_return = _annualized_return(total_return, periods, periods_per_year)
    bench_ann_return = _annualized_return(bench_total_return, periods, periods_per_year)

    strategy_std = _safe_std(strategy_returns)
    benchmark_std = _safe_std(benchmark_returns)
    ann_vol = strategy_std * math.sqrt(periods_per_year)
    bench_ann_vol = benchmark_std * math.sqrt(periods_per_year)

    downside_std = _safe_std(strategy_returns[strategy_returns < 0.0])
    benchmark_downside_std = _safe_std(benchmark_returns[benchmark_returns < 0.0])
    ann_downside = downside_std * math.sqrt(periods_per_year)
    bench_ann_downside = benchmark_downside_std * math.sqrt(periods_per_year)

    sharpe = _safe_div(ann_return - rf_annual, ann_vol)
    bench_sharpe = _safe_div(bench_ann_return - rf_annual, bench_ann_vol)
    sortino = _safe_div(ann_return - rf_annual, ann_downside)
    bench_sortino = _safe_div(bench_ann_return - rf_annual, bench_ann_downside)

    max_drawdown = float(ts["drawdown"].min())
    bench_max_drawdown = float(ts["benchmark_drawdown"].min())
    calmar = _safe_div(ann_return, abs(max_drawdown))
    bench_calmar = _safe_div(bench_ann_return, abs(bench_max_drawdown))

    win_rate = float((strategy_returns > 0.0).mean())
    bench_win_rate = float((benchmark_returns > 0.0).mean())

    covariance = float(np.cov(strategy_returns, benchmark_returns, ddof=1)[0, 1]) if periods > 1 else 0.0
    benchmark_var = float(np.var(benchmark_returns, ddof=1)) if periods > 1 else 0.0
    beta = _safe_div(covariance, benchmark_var)
    alpha_step = (float(strategy_returns.mean()) - rf_step) - beta * (float(benchmark_returns.mean()) - rf_step)
    alpha_annualized = float(alpha_step * periods_per_year)
    info_ratio = _safe_div(float(excess_returns.mean()) * math.sqrt(periods_per_year), _safe_std(excess_returns))

    turnover_units_abs = float(np.abs(ts["position_units"].diff().fillna(0.0)).sum())
    exposure_mean = float(ts["exposure_ratio"].mean())
    exposure_max = float(ts["exposure_ratio"].max())
    trade_count = int(len(trades))
    max_drawdown_duration = _max_drawdown_duration(ts["drawdown"])
    bench_max_drawdown_duration = _max_drawdown_duration(ts["benchmark_drawdown"])

    action_distribution = {
        name: (int(count), float(count / periods))
        for name, count in action_counter.items()
    }

    summary = {
        "run": {
            "algo": algo,
            "checkpoint": str(checkpoint_path),
            "start_index": int(start_index),
            "trading_period": getattr(config.env, "trading_period", None),
            "num_steps": int(periods),
            "start_timestamp": ts.index[0].isoformat(),
            "end_timestamp": ts.index[-1].isoformat(),
            "device": resolved_device,
            "seed": int(seed),
            "epsilon": float(args.epsilon),
            "stochastic_policy": bool(args.stochastic_policy),
        },
        "performance": {
            "reward_return": float(reward_return),
            "total_return": total_return,
            "total_return_pct": total_return * 100.0,
            "annualized_return": ann_return,
            "annualized_return_pct": ann_return * 100.0,
            "annualized_volatility": ann_vol,
            "annualized_volatility_pct": ann_vol * 100.0,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100.0,
            "max_drawdown_duration_steps": int(max_drawdown_duration),
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100.0,
            "alpha_annualized": alpha_annualized,
            "beta": beta,
            "information_ratio": info_ratio,
            "exposure_mean": exposure_mean,
            "exposure_mean_pct": exposure_mean * 100.0,
            "exposure_max": exposure_max,
            "exposure_max_pct": exposure_max * 100.0,
            "turnover_units_abs": turnover_units_abs,
            "trade_count": int(trade_count),
        },
        "benchmark": {
            "total_return": bench_total_return,
            "total_return_pct": bench_total_return * 100.0,
            "annualized_return": bench_ann_return,
            "annualized_return_pct": bench_ann_return * 100.0,
            "annualized_volatility": bench_ann_vol,
            "annualized_volatility_pct": bench_ann_vol * 100.0,
            "sharpe": bench_sharpe,
            "sortino": bench_sortino,
            "calmar": bench_calmar,
            "max_drawdown": bench_max_drawdown,
            "max_drawdown_pct": bench_max_drawdown * 100.0,
            "max_drawdown_duration_steps": int(bench_max_drawdown_duration),
            "win_rate": bench_win_rate,
            "win_rate_pct": bench_win_rate * 100.0,
        },
        "settings": {
            "risk_free_rate": rf_annual,
            "periods_per_year": periods_per_year,
            "rolling_window": int(args.rolling_window),
        },
        "action_distribution": action_distribution,
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    figures = {
        "equity_vs_benchmark": figures_dir / "equity_vs_benchmark.png",
        "drawdown": figures_dir / "drawdown.png",
        "rolling_sharpe": figures_dir / "rolling_sharpe.png",
        "monthly_heatmap": figures_dir / "monthly_returns_heatmap.png",
        "return_distribution": figures_dir / "return_distribution.png",
        "position_action": figures_dir / "position_action_trace.png",
    }
    _plot_equity(ts, figures["equity_vs_benchmark"])
    _plot_drawdown(ts, figures["drawdown"])
    _plot_rolling_sharpe(ts, max(2, int(args.rolling_window)), periods_per_year, figures["rolling_sharpe"])
    _plot_monthly_heatmap(ts, figures["monthly_heatmap"])
    _plot_return_distribution(ts, figures["return_distribution"])
    _plot_position_and_action(ts, figures["position_action"])

    summary_path = run_dir / "tear_sheet_summary.json"
    timeseries_path = run_dir / "tear_sheet_timeseries.csv"
    trades_path = run_dir / "tear_sheet_trades.csv"
    markdown_path = run_dir / "tear_sheet.md"
    html_path = run_dir / "tear_sheet.html"

    ts_reset = ts.reset_index().rename(columns={"timestamp": "date"})
    ts_reset.to_csv(timeseries_path, index=False)
    trades_reset = trades.reset_index().rename(columns={"timestamp": "date"})
    trades_reset.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2))

    figures_rel = {name: str(path.relative_to(run_dir)) for name, path in figures.items()}
    markdown_path.write_text(_build_markdown_report(summary, figures_rel))
    html_path.write_text(_build_html_report(summary, figures_rel))

    print(f"Backtest tear sheet generated at: {run_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Timeseries CSV: {timeseries_path}")
    print(f"Trades CSV: {trades_path}")
    print(f"Markdown report: {markdown_path}")
    print(f"HTML report: {html_path}")


if __name__ == "__main__":
    main()
