# Base D3QN Trading (Capital-Aware RL Research Workspace)

This repository is a reproducible RL trading research workspace with three algorithms on a shared market pipeline:
- D3QN (discrete actions)
- PPO (continuous actions)
- SAC (continuous actions)

## What Changed (Current Phase)

The current branch consolidates the project from legacy notebook behavior into a capital-aware and risk-controlled research stack.

1. Capital-aware environment
- Added explicit `cash / position / equity` accounting to trading environments.
- Added transaction cost and slippage handling at each trade.
- Added optional bankruptcy guard via `min_equity_ratio` and `stop_on_bankruptcy`.

2. D3QN stability upgrades
- Added Prioritized Experience Replay (PER): `per_enabled`, `per_alpha`, `per_beta_start`, `per_beta_steps`, `per_eps`.
- Added n-step TD targets: `n_step` support in agent and trainer.
- Fixed truncation handling: time-limit episode boundaries are no longer forced as terminal in n-step flushing.

3. Discrete action-space upgrades (capital mode)
- Added fractional buy/sell actions through `buy_fractions` and `sell_fractions`.
- Added exposure-based control with `max_exposure_ratio`.
- Added risk-control hooks:
  - `blocked_trade_penalty`
  - `min_hold_steps`
  - `trade_cooldown_steps`
  - `dynamic_exposure_enabled`
  - `dynamic_exposure_vol_window`
  - `dynamic_exposure_min_scale`
  - `dynamic_exposure_strength`

4. Cross-algo benchmark support
- Added PPO and SAC training/evaluation scripts on the same data and environment pipeline.
- Added standardized IS/OOS evaluation outputs in run directories.

5. Tooling and quality
- Added backtest tear sheet generator: `scripts/backtest_report.py`.
- Added test entrypoint: `scripts/test.py`.
- Added D3QN PER+n-step behavior tests under `tests/d3qn/`.

## Repository Layout

- `rl/algos/d3qn/`: D3QN agent, replay buffer, trainer, networks.
- `rl/algos/ppo/`: PPO trainer/networks.
- `rl/algos/sac/`: SAC trainer/networks/replay.
- `rl/envs/`: trading environments and builders.
- `configs/`: reproducible config presets.
- `scripts/`: training/evaluation/report/testing entrypoints.
- `runs/`: curated run artifacts.
- `reports/`: progress and experiment logs.

## Setup

Python 3.10+ and `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

Use `.venv/bin/python` for all commands.

## Key Training Commands

D3QN legacy baseline:

```bash
./.venv/bin/python scripts/train.py \
  --config configs/default.yaml \
  --run-name d3qn_default
```

D3QN capital-aware (6-action stable config, PER+n-step enabled):

```bash
./.venv/bin/python scripts/train.py \
  --config configs/d3qn_signature_capital_6act_stable.yaml \
  --run-name d3qn_capital_6act_stable
```

PPO:

```bash
./.venv/bin/python scripts/train_ppo.py \
  --config configs/ppo_signature.yaml \
  --run-name ppo_continuous
```

SAC:

```bash
./.venv/bin/python scripts/train_sac.py \
  --config configs/sac_signature.yaml \
  --run-name sac_continuous
```

## Evaluation

D3QN:

```bash
./.venv/bin/python scripts/eval.py \
  --config runs/<run_name>/config_resolved.yaml \
  --checkpoint runs/<run_name>/checkpoints/checkpoint_latest.pt \
  --output-dir runs/<run_name>/eval_custom
```

PPO:

```bash
./.venv/bin/python scripts/eval_ppo.py \
  --config runs/<run_name>/config_resolved.yaml \
  --checkpoint runs/<run_name>/checkpoints/checkpoint_latest.pt \
  --output-dir runs/<run_name>/eval_custom
```

SAC:

```bash
./.venv/bin/python scripts/eval_sac.py \
  --config runs/<run_name>/config_resolved.yaml \
  --checkpoint runs/<run_name>/checkpoints/checkpoint_latest.pt \
  --output-dir runs/<run_name>/eval_custom
```

## Backtest Tear Sheet

```bash
./.venv/bin/python scripts/backtest_report.py \
  --checkpoint runs/<run_name>/checkpoints/checkpoint_latest.pt \
  --output-dir runs/<run_name>/tear_sheet
```

## Tests

```bash
./.venv/bin/python scripts/test.py
```

## Active Curated Runs

The `runs/` directory has been curated to keep only key comparison runs used in current reports.
See `reports/experiments.md` for metric tables and interpretation.

## Reporting

- Progress timeline: `reports/progress.md`
- Quantitative experiment log: `reports/experiments.md`
