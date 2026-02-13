# Base D3QN Trading

Training-first refactor of a D3QN-style trading baseline.
Core code lives in `rl/`, with CLI entrypoints in `scripts/`.

## Repository Structure

- `rl/algos/d3qn/`: agent, schedules, trainer, replay buffer, networks.
- `rl/envs/`: trading environment and environment builders.
- `scripts/train.py`: training entrypoint.
- `scripts/eval.py`: checkpoint evaluation entrypoint.
- `configs/`: experiment configurations.
- `runs/`: output artifacts per run.
- `reports/`: progress and experiment logs.
- `legacy/`: notebook-derived reference code.

## Setup

Use Python 3.10+ and `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

Prefer `.venv/bin/python` for all commands.

## Training

Basic run:

```bash
./.venv/bin/python scripts/train.py --config configs/default.yaml --run-name d3qn_profit --output-dir runs
```

Signature observation example:

```bash
./.venv/bin/python scripts/train.py --config configs/test_signature.yaml --run-name test_signature --reward sr_enhanced --output-dir runs
```

Useful overrides:

- `--reward {profit,sr,sr_enhanced}`
- `--num-episodes`, `--total-steps`
- `--device`
- `--override key=value` (repeatable), for example:
  - `--override train.eval_interval=20`
  - `--override train.eval_episodes=50`

### Training Artifacts

Each run writes to `runs/<run_name>/`:

- `config_resolved.yaml`: full resolved config snapshot.
- `metrics.csv`: episode-level training metrics (`reward_return`, epsilon, loss, q).
- `checkpoints/`: periodic and latest checkpoints.
- `eval_history.csv`: periodic in-training eval snapshots (if enabled).
- `run.log`: logger output.
- `tensorboard/`: TensorBoard event files.

## Evaluation

Evaluate a checkpoint:

```bash
./.venv/bin/python scripts/eval.py \
  --config runs/<run_name>/config_resolved.yaml \
  --checkpoint runs/<run_name>/checkpoints/checkpoint_latest.pt \
  --output-dir runs/<run_name>
```

### Eval Outputs

- `eval_summary.json`: aggregate metrics.
- `eval_episodes.csv`: per-episode rows (when `eval.save_per_episode=true`).

Key metric naming:

- `reward_return`: sum of step rewards over one episode.
- `return_rate`: portfolio return computed as `(equity_end / equity_start) - 1`.
- `*_pct` fields in `eval_summary.json`: percentage version of return rates.
- `win_rate`: ratio of episodes with positive `reward_return` (reward-based win rate).

Diagnostics:

- `initial_state_none_episodes`
- `zero_step_episodes`

These help detect evaluation loops that fail to run steps.

## Recent Behavioral Notes

- `sr_enhanced` reward is supported end-to-end (`train.py`, `eval.py`, env).
- Epsilon now decays linearly over global environment steps (not reset each episode).
- In-training periodic eval uses fixed windows by seed and logs to `eval_history.csv`.
- `trainer.evaluate()` and `scripts/eval.py` now use consistent `trading_period` handling.

## Reports

- Progress timeline: `reports/progress.md`
- Experiment metrics log: `reports/experiments.md`

Update both when adding new training/evaluation changes.

## Legacy and Notebooks

`legacy/` keeps notebook-era reference code.
Use `rl/` + `scripts/` for current experiments and reproducible runs.
