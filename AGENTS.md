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
- `legacy/src/` — deprecated notebook-derived code kept for reference
- `notebooks/`
  - `main.ipynb` — example workflow (load data → create env/agent → test, optional train)
  - `train_tests.ipynb` — experiments/tests notebook
- `data/`
  - `Bitcoin History 2010-2024.csv`

> Note: The repo currently contains `__pycache__/` and `.DS_Store` entries. Do not add more such
> machine-specific or generated files in future changes.

## Environment & setup

### Python

The repository includes Python 3.10 bytecode artifacts, so Python **3.10+** is a safe default.

### Dependencies

A baseline `requirements.txt` is provided. Install the libraries used by notebooks and
the core `rl/` code.

Minimum (observed in notebooks):
- `pandas`
- `prettytable`

Commonly needed for RL experiments in this repo:
- `numpy`
- `matplotlib`

Model/backbone dependencies:
- Check `rl/algos/d3qn/networks.py` and `rl/algos/d3qn/agent.py` imports and install the required deep learning framework
  (e.g., PyTorch or TensorFlow), plus any utilities they rely on.

Suggested setup (example):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install -r requirements.txt
# If you swap deep learning frameworks, update requirements accordingly.
