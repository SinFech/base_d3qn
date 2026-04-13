# Step 16 - Sweep `logsig.degree` on the Frozen Baseline Embedding

## Goal

Test whether changing the signature truncation level alone can produce a stronger `f1` specialist branch
than the current baseline `degree=3` setting.

This step stays inside the current specialist interpretation:

- only `f1`
- only the two primary `f1` metrics
- allow candidate-side cherry-pick interpretation only after passing the short gate
- keep the official repository baseline reference fixed at the official `f1` `5`-seed average

## Scope

This step should:

- keep the baseline path embedding fixed:
  - `log_price`
  - `log_return`
  - `rolling_vol(window=5)`
- keep the frozen `ConvDuelingDQN` encoder
- vary only `signature.logsig.degree`
- measure whether lower or higher truncation depth creates better `f1` behavior

This step should not:

- change the embedding channels
- change the rolling-vol horizon
- switch to `MLPDuelingDQN`
- reopen `f2`, `f3`, or aggregate OOS criteria
- reinterpret the result as a repository-wide default change

## Why This Step Exists

So far, most follow-up work changed:

- embedding composition
- replacement channels
- rolling-vol horizon
- encoder family

But the signature truncation level itself has not yet been tested under the current `f1` specialist line.

Step 1 already included `C6_deg4`, but that candidate was filtered under the old `f3`-first funnel.
That means the repository still lacks a clean answer to this narrower question:

- if the baseline embedding stays fixed, which `logsig.degree` is best for `f1`?

## Inputs

- `docs/signature/plan/step1.md`
- `docs/signature/plan/step9.md`
- `docs/signature/plan/step15.md`
- `docs/signature/plan/baseline_metrics.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/folds_signature_step9_f1.json`
- `rl/features/signature.py`
- `rl/envs/wrappers.py`
- `scripts/walk_forward_protocol.py`

## Official External Reference

Final reporting in this step must keep the same official baseline reference:

- source:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- fixed `f1` baseline:
  - `f1 oos_sharpe_mean = 0.8904662600702572`
  - `f1 oos_return_pct_mean = 85.86166745250878`

## Local Family Control

Within this sweep, the family control is:

- `L3_deg3`

Reason:

- it matches the current frozen baseline degree
- it provides the correct within-family same-seed control for short screening

## Frozen Candidate Set

All Step 16 candidates must keep the same path embedding and the same encoder.
Only `signature.logsig.degree` may change.

| Candidate ID | `logsig.degree` | Purpose |
|---|---:|---|
| `L1_deg1` | `1` | low-order test |
| `L2_deg2` | `2` | lighter interaction-order test |
| `L3_deg3` | `3` | family control, identical to current baseline degree |
| `L4_deg4` | `4` | high-order test, reopens the old `C6_deg4` idea under `f1` scope |

Not included:

- `degree <= 0`
- `degree >= 5`

Reason for excluding `degree >= 5`:

- observation size grows too quickly relative to the current baseline family
- this step is intended to answer the local degree question before opening a larger-capacity branch

## Config Rule

Every Step 16 config must:

- start from the frozen baseline config family
- keep the same account features
- keep the same path embedding
- keep the same `ConvDuelingDQN` backbone
- change only:
  - `env.obs.signature.logsig.degree`

## Protocol

### `S16_short_f1_degree`

Run one short `f1`-only degree sweep:

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

- `L1_deg1`
- `L2_deg2`
- `L3_deg3`
- `L4_deg4`

Short promotion rule:

- promote only candidate-seed pairs where the same short-run seed beats the matched `L3_deg3` seed on both:
  - `f1 oos_sharpe`
  - `f1 oos_return_pct`

### `S16_single_seed_full`

For every promoted candidate-seed pair, run:

- one single-seed full `f1` evaluation for the promoted candidate
- one same-seed full `L3_deg3` control if that exact seed is not already available

Full budget family:

- config default
- effective default in this family:
  - `train.num_episodes = 260`
  - `train.max_total_steps = 100000`

## Final Reporting Rule

Step 16 should report final winners using this interpretation:

1. candidate full-run seed must beat the official baseline `f1` `5`-seed average on both:
   - `OOS Sharpe`
   - `OOS Return %`
2. if one config has multiple winning seeds, keep only the best cherry-pick seed
3. choose that best seed in this order:
   - higher `f1 OOS Sharpe`
   - then higher `f1 OOS Return %`

Step 16 should also report the full-family mean ranking when available, even if no config clears the
official two-metric final rule.

## Deliverables

- one planning note:
  - `docs/signature/plan/step16.md`
- one config family:
  - `configs/signature_step16/`
- one short-screen result note:
  - `docs/signature/plan/step16_degree_short_results.md`
- one optional full result note:
  - `docs/signature/plan/step16_degree_single_seed_full_results.md`
- one short-run output root:
  - `runs/step16_f1_degree_short/`
- one optional full-run output root:
  - `runs/step16_f1_degree_full/`

## Outcome Type

Even if Step 16 finds a stronger candidate, the interpretation remains:

- evidence about how signature truncation level affects `f1`

not:

- a repository-wide default decision
