# Base D3QN Trading

Training-first refactor of the D3QN trading baseline. Core logic lives in `rl/`,
with simple script entrypoints in `scripts/` and notebooks under `notebooks/`.

## Quickstart

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## Train

```bash
python scripts/train.py --config configs/default.yaml --run-name d3qn_profit
```

Artifacts are written to `runs/<run_name>/`:
- `runs/<run_name>/checkpoints/`
- `runs/<run_name>/metrics.csv`
- `runs/<run_name>/tensorboard/`
- `runs/<run_name>/config_resolved.yaml`

## Evaluate

```bash
python scripts/eval.py --checkpoint runs/<run_name>/checkpoints/checkpoint_latest.pt
```

## Smoke test

```bash
python scripts/smoke_test.py
```

## Configs and reproducibility

- Update `configs/default.yaml` for reproducible experiments.
- Use `--seed`, `--num-episodes`, `--total-steps`, and other CLI overrides to
  create variations without editing code.
- Each run writes a resolved config snapshot next to the logs.

## Signature observations (logsignature)

Install `pysiglib`, then set `env.obs.type: signature` and `model.type: mlp_dueling`
in your config (see `configs/signature_smoke.yaml` for a minimal example).

## Notebooks

Notebooks are consumers of the `rl/` modules. Use them to explore training and
analysis, not as the primary source of training logic.

## Legacy code

The original notebook-derived code is preserved under `legacy/` for reference.
Use the new `rl/` modules and `scripts/` entrypoints for current experiments.
