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
- `runs/`: curated run summaries only.
- `reports/`: progress and experiment logs.

## Setup

Python 3.10+ and `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

Use `.venv/bin/python` for all commands.

## Savio Slurm

A reusable Savio GPU batch script is available at `scripts/slurm/train_d3qn_savio.sbatch`.
Submit it from the repository root so Slurm writes logs into `./logs/`.

For local Savio password generation without committing secrets, use `scripts/savio_pwd.py`.
The script reads `SAVIO_PIN` and `SAVIO_OTP_URI` from environment variables or a local-only file such as `.secrets/savio.env`.

Example:

```bash
mkdir -p .secrets
cp configs/savio.env.example .secrets/savio.env
python scripts/savio_pwd.py --config .secrets/savio.env
```

Example GPU smoke run:

```bash
RUN_NAME=d3qn_sig_gpu_smoke \
NUM_EPISODES=2 \
TOTAL_STEPS=500 \
EVAL_INTERVAL=0 \
sbatch scripts/slurm/train_d3qn_savio.sbatch
```

Common environment overrides:
- `CONFIG_PATH`: training config path. Default: `configs/d3qn_signature_capital_6act_per06_n3_worstfold_gpu.yaml`
- `TRAIN_DEVICE`: `cuda` or `cpu` for `scripts/train.py`
- `SIGNATURE_DEVICE`: device for `env.obs.signature.torch.device`
- `RUN_NAME`: run-name prefix passed to `scripts/train.py`
- `NUM_EPISODES`, `TOTAL_STEPS`, `LOG_INTERVAL`, `OUTPUT_DIR`
- `EXTRA_OVERRIDES`: semicolon-delimited extra overrides, for example `run.seed=43;train.eval_interval=0`

## Retention Policy

This repository keeps reproducibility-critical configs and only lightweight run summaries in git.

- Kept in `runs/`:
  - `summary_by_algo.csv`
  - `summary_by_algo_fold.csv`
- Ignored in `runs/`:
  - checkpoints
  - tensorboard logs
  - per-run metrics/logs
  - per-run eval directories
  - resolved run configs

This keeps historical comparison points without committing heavyweight training artifacts.

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

## Standardized Walk-Forward Protocol (3 Folds x Multi-Seed)

Default fold schedule in `configs/folds_rolling_long_oos.json`:
- `f1`: train `2014-01-01 ~ 2018-12-31`, test `2019-01-01 ~ 2022-12-31`
- `f2`: train `2015-01-01 ~ 2019-12-31`, test `2020-01-01 ~ 2023-12-31`
- `f3`: train `2016-01-01 ~ 2020-12-31`, test `2021-01-01 ~ 2024-02-09`

Run the full standardized protocol (default: seeds `42,43,44,45,46`):

```bash
./.venv/bin/python scripts/walk_forward_protocol.py \
  --algos ppo d3qn \
  --output-dir runs/wf_3fold_5seed
```

Validate the plan only (no training):

```bash
./.venv/bin/python scripts/walk_forward_protocol.py --dry-run
```

Key protocol defaults:
- `train_split=1.0` in this script (to avoid shrinking each sampled `trading_period` window to 80%).
- Eval start windows are sampled with replacement (repeat is allowed) by default.
- Each run name includes a timestamp `run_tag` by default to avoid overwriting prior runs.
- Use `--run-tag <tag> --skip-existing` to resume an interrupted batch safely.

Outputs:
- Per-run artifacts: `runs/<output_dir>/<algo>_<fold>_s<seed>_<run_tag>/`
- Aggregated table: `runs/<output_dir>/results.csv`
- Algo+fold table: `runs/<output_dir>/summary_by_algo_fold.csv`
- Algo-level summary: `runs/<output_dir>/summary_by_algo.csv`

## Tests

```bash
./.venv/bin/python scripts/test.py
```

## Active Curated Runs

The `runs/` directory is curated to keep only key comparison summaries used in current reports.

Retained current-summary directories:
- `runs/batch_wf_rolling_long_oos_repeat`: PPO baseline and D3QN baseline under the same rolling walk-forward protocol.
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`: earlier high-return D3QN milestone (`6-action`, `PER=0.6`, `n_step=3`).
- `runs/wf_d3qn_sellfrac_wf_sellfrac_20260308_125429`: current D3QN mainline (`6-action sell_fractions`).
- `runs/wf_d3qn_7act_sellfrac_wf_7act_sellfrac_20260309_011013`: `7-action` comparison run.
- `runs/wf_d3qn_8act_sellfrac_wf_8act_sellfrac_20260309_122333`: `8-action` comparison run.

See `reports/experiments.md` for metric tables and interpretation.

## Retained Key Configs

Historical and current key configs are intentionally kept under `configs/`:

- Baselines:
  - `configs/baseline.yaml`
  - `configs/ppo_signature.yaml`
  - `configs/sac_signature.yaml`
- D3QN milestones:
  - `configs/d3qn_signature_capital_6act_stable.yaml`
  - `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
  - `configs/d3qn_signature_capital_6act_per06_n3_sellfrac_gpu.yaml`
  - `configs/d3qn_signature_capital_7act_per06_n3_sellfrac_gpu.yaml`
  - `configs/d3qn_signature_capital_8act_per06_n3_sellfrac_gpu.yaml`
- Fair-comparison risk-budget variants:
  - `configs/d3qn_signature_capital_6act_per06_n3_sellfrac_e07_bp001_gpu.yaml`
  - `configs/d3qn_signature_capital_8act_per06_n3_sellfrac_e07_bp001_gpu.yaml`
  - `configs/d3qn_signature_capital_6act_per06_n3_sellfrac_e06_bp002_gpu.yaml`
  - `configs/d3qn_signature_capital_8act_per06_n3_sellfrac_e06_bp002_gpu.yaml`
- Protocol definition:
  - `configs/folds_rolling_long_oos.json`

## Reporting

- Progress timeline: `reports/progress.md`
- Quantitative experiment log: `reports/experiments.md`
