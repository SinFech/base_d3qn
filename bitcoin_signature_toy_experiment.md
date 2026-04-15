# Bitcoin Signature Toy Experiment

## Purpose

This note summarizes the Bitcoin-aligned toy experiment implemented in:

- `scripts/signature_toy_experiment_bitcoin.py`

The goal is not to show trading alpha. The goal is to verify, at the representation level, that the current signature pipeline can encode within-window temporal structure on real Bitcoin market windows from the repository dataset.

Concretely, the experiment asks:

- if two samples share the same real Bitcoin price path
- and differ only in how volume is distributed across the window
- can a model using signature features separate them better than a model without signature features?

## Task Design

### Data source

- dataset: `data/Bitcoin History 2010-2024.csv`
- loader: `rl.envs.make_env.load_price_data(...)`
- train date range: `2014-01-01` to `2020-12-31`
- test date range: `2021-01-01` to `2024-02-09`
- window size: `24`
- source windows per split: `256`

Each source sample is a real Bitcoin OHLCV window sampled from the repository dataset.

### Label construction

The task uses paired counterfactual variants built from the same real window.

For each source window:

- keep `Close`, `High`, and `Low` unchanged
- keep total window volume unchanged
- modify only the time profile of `Volume`

Two classes are created:

- `early_volume`
  - volume is biased toward the first half of the window
- `late_volume`
  - volume is biased toward the second half of the window

The current default perturbation is a partial mixture, not a full rewrite:

- `volume_mix_alpha = 0.45`

This means the modified volume path is:

- `55%` original normalized volume shape
- `45%` target early or late profile

This makes the task non-trivial:

- a non-signature model can extract some useful signal from the raw window
- but the signature model still has a clear advantage

## Feature Sets and Probe Model

The script compares three feature families.

### 1. `summary`

This is a weak sanity-check baseline built from coarse aggregated statistics:

- final log return
- mean log return
- std of log return
- max close
- min close
- close range
- total volume
- mean volume
- volume std

This baseline does not explicitly encode where volume occurred inside the window.

### 2. `raw_window`

This is the main non-signature baseline.

It flattens the full real-valued market window:

- `Close`
- `High`
- `Low`
- `Volume`

For `24` bars and `4` channels, the input dimension is:

- `96`

### 3. `signature`

This uses the repository signature pipeline directly:

- `PathBuilder`
- `LogSigTransformer`

Current embedding:

- `log_price`
- `log_return`
- `normalized_cumulative_volume`
- `high_low_range`

Current transformer settings:

- degree `2`
- `time_aug = true`
- `lead_lag = false`

The resulting feature dimension is:

- `15`

### Probe model

The current default classifier is a small MLP:

- input
- linear `-> 32`
- `ReLU`
- linear `-> 32`
- `ReLU`
- linear `-> 1`

Training settings:

- optimizer: `Adam`
- loss: `BCEWithLogitsLoss`
- epochs: `400`
- learning rate: `0.001`
- seed: `17`

The same probe architecture is used for all three feature families.

## Default Result

Default command:

```bash
.venv/bin/python scripts/signature_toy_experiment_bitcoin.py
```

Observed output:

| Feature set | Dim | Train accuracy | Test accuracy |
|---|---:|---:|---:|
| `summary` | `9` | `0.5625` | `0.4961` |
| `raw_window` | `96` | `0.8652` | `0.6797` |
| `signature` | `15` | `0.9512` | `0.9141` |

Class summary from the same run:

| Class | Mean total volume | Mean first-half share | Mean second-half share |
|---|---:|---:|---:|
| `early_volume` | `655264622.2656` | `0.5399` | `0.4601` |
| `late_volume` | `655264622.2656` | `0.4617` | `0.5383` |

Key read:

- `summary` is essentially at chance
- `raw_window` learns a moderate signal and reaches about `0.68` test accuracy
- `signature` reaches about `0.91` test accuracy on the same task

## Interpretation

The main conclusion is:

- the current signature feature stack is encoding within-window volume timing structure on real Bitcoin windows
- and it is doing so more effectively than a non-signature model using the raw flattened window

This is exactly the kind of result a mechanism experiment should produce.

What the result does show:

- signature features contain linearly or near-linearly accessible information about when volume accumulates within the window
- the `normalized_cumulative_volume` path channel is not decorative
- the signature transform is preserving useful temporal geometry rather than only coarse scale information

What the result does not show:

- it does not prove trading profitability
- it does not prove out-of-sample market prediction skill
- it does not prove that the current D3QN policy will improve by the same margin

This experiment validates representation power, not downstream strategy performance.

## Why the Result Is Reasonable

The result is strong but not pathological.

Earlier versions of the toy experiment used:

- a linear probe
- a fully rewritten early-vs-late volume profile

That setup produced an overly clean separation, including cases near:

- non-signature baseline `~0.50`
- signature `~1.00`

The current version is deliberately harder and more realistic:

- the probe is a small MLP, not a trivial linear head
- the volume perturbation is partial, not absolute
- the raw-window baseline now captures some signal

That is why the present default is a more credible regime:

- `raw_window ~ 0.68`
- `signature ~ 0.91`

## Main Writing Angle

For later writing, the clean claim is:

> On real Bitcoin OHLCV windows from the project dataset, a controlled early-vs-late volume-timing task is only moderately solvable from the raw flattened window, but becomes highly solvable after the repository signature transform. This supports the view that the signature pipeline is encoding within-window temporal structure rather than merely coarse aggregate statistics.

## Reproducibility

Run the experiment:

```bash
.venv/bin/python scripts/signature_toy_experiment_bitcoin.py
```

Run the tests:

```bash
.venv/bin/python -m pytest tests/signature/test_signature_toy_experiment_bitcoin.py tests/signature/test_signature_toy_experiment.py tests/signature/test_signature_features.py
```

Current status at the time of writing:

- the script was run successfully with the default configuration above
- the related signature tests passed: `9 passed`
