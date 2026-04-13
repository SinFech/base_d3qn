# Step 15 - Validate the Encoder Bottleneck Hypothesis with `MLPDuelingDQN`

## Goal

Test whether the current `ConvDuelingDQN` encoder is suppressing embedding-specific `f1` signal
that appears only as narrow specialist wins under the existing signature setup.

This step validates one concrete hypothesis:

- some promising signature embeddings are not failing because the embedding is bad
- they are failing because the current flat-logsignature observation is a poor fit for the convolutional encoder

## Scope

This step should:

- stay inside the `f1` specialist line
- keep the path embeddings fixed and change only the D3QN encoder family
- use the already-implemented `MLPDuelingDQN` model type
- compare only a narrow set of embeddings that already showed some specialist evidence

This step should not:

- redesign the signature feature extractor
- change reward shaping, action space, or risk controls
- add a new dual-branch encoder in this step
- widen candidate scope beyond the targeted validation set

## Primary Metrics

All execution and reporting in this step should stay centered on:

- `f1 oos_sharpe`
- `f1 oos_return_pct`

## Why This Step Exists

The current signature observation is:

- a flattened logsignature vector
- followed by appended account features

The current D3QN default still feeds that vector through `ConvDuelingDQN`.

That may be a poor inductive bias because:

- neighboring logsignature coordinates do not necessarily define a meaningful local spatial structure
- added or replaced embedding channels may alter coordinate layout without giving the conv stack a better signal model

Steps 11 and 12 found two notable specialist directions:

- `D3_return_vol5`
- `R5_volprof_for_return`

Both produced same-seed full-run wins, but neither has yet been tested under a non-convolutional encoder.

## Inputs

- `docs/signature/plan/step11_reduction_single_seed_full_results.md`
- `docs/signature/plan/step12_replacement_single_seed_full_results.md`
- `docs/signature/plan/step14.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/signature_step11/d3_return_vol5.yaml`
- `configs/signature_step12/r5_volprof_for_return.yaml`
- `configs/folds_signature_step9_f1.json`
- `rl/algos/d3qn/networks.py`
- `rl/algos/d3qn/trainer.py`
- `scripts/walk_forward_protocol.py`

## Frozen Candidate Set

Encoder family for this step:

- `model.type = mlp_dueling`

All other non-embedding settings remain aligned with the frozen baseline recipe.

| Candidate ID | Base embedding | Purpose |
|---|---|---|
| `MLP_baseline` | `log_price + log_return + rolling_vol(window=5)` | encoder control |
| `MLP_D3_return_vol5` | `log_return + rolling_vol(window=5)` | validate the strongest reduction-style specialist signal under MLP |
| `MLP_R5_volprof_for_return` | `log_price + rolling_vol(window=5) + normalized_cumulative_volume` | validate the strongest replacement-style specialist signal under MLP |

## Protocol

### `S15_short_f1_mlp`

Run one short `f1`-only validation batch:

- fold protocol:
  - `configs/folds_signature_step9_f1.json`
- fold:
  - `f1` only
- seeds:
  - `42,43,44`
- short budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- eval episodes:
  - `50`

Candidate batch:

- `MLP_baseline`
- `MLP_D3_return_vol5`
- `MLP_R5_volprof_for_return`

Short promotion rule:

- promote candidate-seed pairs only when the same short-run seed beats the matched `MLP_baseline` seed on both:
  - `f1 oos_sharpe`
  - `f1 oos_return_pct`

### `S15_full_f1_mlp`

If at least one candidate-seed pair is promoted:

- run full `f1` `5`-seed evaluation for:
  - `MLP_baseline`
  - every promoted candidate config

Full budget family:

- config default
- effective default in this family:
  - `train.num_episodes = 260`
  - `train.max_total_steps = 100000`

## Final Reporting Rule

Step 15 should answer two questions separately:

1. relative to `MLP_baseline`, do `D3` or `R5` become stronger and more stable?
2. relative to the official repository baseline `f1` `5`-seed average, does any `MLP` candidate produce a new two-metric full-run winner?

If a full-stage config has multiple winning seeds against the official baseline:

- keep only the best seed
- rank by:
  1. higher `f1 OOS Sharpe`
  2. then higher `f1 OOS Return %`

## Deliverables

- one planning note:
  - `docs/signature/plan/step15.md`
- one config family:
  - `configs/signature_step15/`
- one short-screen result note:
  - `docs/signature/plan/step15_mlp_short_results.md`
- one optional full result note:
  - `docs/signature/plan/step15_mlp_full_results.md`
- one short-run output root:
  - `runs/step15_f1_mlp_short/`
- one optional full-run output root:
  - `runs/step15_f1_mlp_full/`

## Outcome Type

Even if Step 15 finds a stronger candidate, the interpretation remains:

- evidence about whether the current encoder is the bottleneck for `f1` signature specialists

not:

- a repository-wide default change
