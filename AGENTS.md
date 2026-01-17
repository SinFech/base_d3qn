# AGENTS.md

This file provides practical instructions for AI coding agents (and humans) working in this repository.
Keep changes small, reproducible, and aligned with the existing RL/trading workflow.
All code and comments must be written in English.

## Repository overview

This repo is a small baseline deep reinforcement learning project centered around a D3QN-style agent
applied to a trading environment (example uses Bitcoin historical price data). Core code lives in `rl/`,
with notebooks under `notebooks/` and a CSV dataset under `data/`. The original notebook-derived code is
kept under `legacy/src/` for reference only.

### Key paths

- `rl/`
  - `algos/d3qn/` — agent, networks, replay buffer, trainer
  - `envs/` — trading environment and data helpers
  - `utils/` — logging, seeding, checkpoint helpers
- `scripts/`
  - `train.py` — training entry point
  - `eval.py` — evaluation entry point
- `legacy/` — deprecated notebook-derived code kept for reference
- `data/`
  - `Bitcoin History 2010-2024.csv`
- `reports/`
  - `progress.md` — running progress log
  - `experiments.md` — experiment results log

> Note: The repo currently contains `__pycache__/` and `.DS_Store` entries. Do not add more such
> machine-specific or generated files in future changes.

## Experiment logging

When running or modifying experiment-related code, review `reports/progress.md` and
`reports/experiments.md` first to stay aligned with the latest progress and results.

## Environment & setup

### Python

The repository includes Python 3.10 bytecode artifacts, so Python **3.10+** is a safe default.

### Dependencies

Dependencies are managed with `uv` via `pyproject.toml` and `uv.lock`. Use `uv` to
add packages and refresh the lockfile.

Suggested setup (example):
```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

Default to using `.venv/bin/python` for running scripts in this repo.

When adding dependencies:
```bash
uv add <package>
uv lock
```
