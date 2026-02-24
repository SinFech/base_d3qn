# Base D3QN Trading

Refactored RL trading workspace with:
- D3QN (discrete actions)
- PPO (continuous actions)
- SAC (continuous actions)
- Capital-constrained environments with explicit `cash/position/equity` accounting

Core implementation lives in `rl/`; runnable entrypoints are in `scripts/`.

## Repository Structure

- `rl/algos/d3qn/`: D3QN agent, replay buffer, trainer, networks.
- `rl/algos/ppo/`: PPO continuous policy, trainer, networks.
- `rl/algos/sac/`: SAC continuous policy, trainer, replay buffer, networks.
- `rl/envs/`: environment builders and env implementations:
  - legacy discrete env
  - `DiscreteCapitalTradingEnvironment`
  - `ContinuousTradingEnvironment`
- `scripts/`: train/eval CLIs for D3QN/PPO/SAC.
- `configs/`: runnable config presets.
- `runs/`: run artifacts.
- `reports/`: progress and experiment logs.
- `legacy/`: notebook-era reference code.

## Setup

Use Python 3.10+ and `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

Use `.venv/bin/python` for commands.

## Environments and Action Modes

`make_env(...)` supports:
- `action_mode=discrete`: legacy discrete env
- `action_mode=discrete_capital`: discrete actions with capital accounting
- `action_mode=continuous`: continuous position target with capital accounting

For `discrete_capital`, you can configure:
- `max_exposure_ratio`: cap on notional exposure ratio
- `buy_fractions`: fractional buy actions
- `sell_fractions`: fractional sell actions

Action count rule:
- If fractional actions are enabled:
  - `agent.action_number = 1 + len(buy_fractions) + len(sell_fractions)`
  - action `0` is always `hold`
- If only `buy_fractions` is provided, one `sell_all` action is added for compatibility.

## Training

### D3QN (default / legacy)

```bash
./.venv/bin/python scripts/train.py \
  --config configs/default.yaml \
  --run-name d3qn_default
```

### D3QN with Capital-Constrained Signature Env

```bash
./.venv/bin/python scripts/train.py \
  --config configs/d3qn_signature_capital.yaml \
  --run-name d3qn_capital
```

### D3QN Fractional Buy/Sell Example

```bash
./.venv/bin/python scripts/train.py \
  --config configs/d3qn_signature_capital.yaml \
  --run-name d3qn_fractional \
  --override env.action_mode=discrete_capital \
  --override env.max_exposure_ratio=0.8 \
  --override env.sell_mode=all \
  --override env.buy_fractions='[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]' \
  --override env.sell_fractions='[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]' \
  --override agent.action_number=17
```

### PPO (continuous)

```bash
./.venv/bin/python scripts/train_ppo.py \
  --config configs/ppo_signature.yaml \
  --run-name ppo_continuous
```

### SAC (continuous)

```bash
./.venv/bin/python scripts/train_sac.py \
  --config configs/sac_signature.yaml \
  --run-name sac_continuous
```

Useful shared overrides:
- `--reward {profit,sr,sr_enhanced}`
- `--num-episodes`, `--total-steps`
- `--device`
- `--override key=value` (repeatable)

## Evaluation

### D3QN

```bash
./.venv/bin/python scripts/eval.py \
  --config runs/<run_name>/config_resolved.yaml \
  --checkpoint runs/<run_name>/checkpoints/checkpoint_latest.pt \
  --output-dir runs/<run_name>/eval_custom
```

### PPO

```bash
./.venv/bin/python scripts/eval_ppo.py \
  --config runs/<run_name>/config_resolved.yaml \
  --checkpoint runs/<run_name>/checkpoints/checkpoint_latest.pt \
  --output-dir runs/<run_name>/eval_custom
```

### SAC

```bash
./.venv/bin/python scripts/eval_sac.py \
  --config runs/<run_name>/config_resolved.yaml \
  --checkpoint runs/<run_name>/checkpoints/checkpoint_latest.pt \
  --output-dir runs/<run_name>/eval_custom
```

## Run Artifacts

Each run under `runs/<run_name>/` typically contains:
- `config_resolved.yaml`
- `metrics.csv`
- `checkpoints/`
- `run.log`
- `tensorboard/`
- `eval_summary.json`, `eval_episodes.csv` (after eval)

## Metric Conventions

- `reward_return`: episode sum of environment rewards.
- `return_rate`: `(equity_end / equity_start) - 1`.
- `*_pct`: percentage representation of return-rate metrics.
- diagnostics include:
  - `initial_state_none_episodes`
  - `zero_step_episodes`

## Reports

- Progress timeline: `reports/progress.md`
- Experiment records: `reports/experiments.md`

Update both when changing training logic or running new experiment batches.
