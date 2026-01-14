# AGENTS.md

This file provides practical instructions for AI coding agents (and humans) working in this repository.
Keep changes small, reproducible, and aligned with the existing RL/trading workflow.
All code and comments must be written in English.

## Repository overview

This repo is a small baseline deep reinforcement learning project centered around a D3QN-style agent
applied to a trading environment (example uses Bitcoin historical price data). Core code lives in `src/`,
with notebooks under `notebooks/` and a CSV dataset under `data/`.

### Key paths

- `src/`
  - `Agent.py` — agent logic (training/testing interface used by the notebooks)
  - `Environment.py` — trading environment used by the agent
  - `model.py` — network/model implementation
  - `utils.py` — plotting/stat helpers used by notebooks
  - `main.py` — script entry point (if used)
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

There is no pinned `requirements.txt` in the repository. Install the libraries used by notebooks and
the core `src/` code.

Minimum (observed in notebooks):
- `pandas`
- `prettytable`

Commonly needed for RL experiments in this repo:
- `numpy`
- `matplotlib`

Model/backbone dependencies:
- Check `src/model.py` and `src/Agent.py` imports and install the required deep learning framework
  (e.g., PyTorch or TensorFlow), plus any utilities they rely on.

Suggested setup (example):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install pandas numpy matplotlib prettytable
# Then install the deep learning framework required by src/model.py (verify imports first).
