# Step 9 `f1`-Only Full Follow-Up

## Scope

This note records the executed `S9_full_f1` follow-up from Step 9.

Only one candidate was promoted from the short screen:

- `F1_hlrange_rv10`

This config combines:

- `high_low_range`
- `rolling_vol.window = 10`

Config source:

- `configs/signature_step9/f1_hlrange_rv10.yaml`

## Protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - `42,43,44,45,46`
- Full budget family:
  - use the config default training budget
  - effective default in this config family:
    - `train.num_episodes = 260`
    - `train.max_total_steps = 100000`
- Eval episodes:
  - `50`

Execution detail:

- The five full runs were launched as parallel single-seed jobs:
  - `runs/step9_f1_full/f1_hlrange_rv10_seed42/`
  - `runs/step9_f1_full/f1_hlrange_rv10_seed43/`
  - `runs/step9_f1_full/f1_hlrange_rv10_seed44/`
  - `runs/step9_f1_full/f1_hlrange_rv10_seed45/`
  - `runs/step9_f1_full/f1_hlrange_rv10_seed46/`
- The five seed-level `results.csv` files were then merged into:
  - `runs/step9_f1_full/f1_hlrange_rv10/`

Baseline control:

- official baseline `f1` result from:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`

## Full `f1` Comparison

| Config | `f1` OOS Sharpe mean | `f1` OOS Return % mean |
|---|---:|---:|
| baseline | `0.8905` | `85.8617` |
| `F1_hlrange_rv10` | `0.9072` | `84.7223` |
| delta (`candidate - baseline`) | `+0.0168` | `-1.1393` |

Candidate dispersion across the five full seeds:

- `f1 oos_sharpe_std = 0.1280`
- `f1 oos_return_pct_std = 28.3101`

## Interpretation

- `F1_hlrange_rv10` preserved the Step 9 Sharpe-improvement signal in the full follow-up.
- It did not become a strict two-metric winner, because `f1` return remained slightly below the official baseline.
- The full result is therefore:
  - stronger on `f1` Sharpe
  - slightly weaker on `f1` Return

This means Step 9 did **not** find a configuration that clearly improves both full-run `f1` primary metrics at the same time.

## Step 9 Decision

Repository interpretation for Step 9:

- Keep the official baseline as the strongest two-metric `f1` reference.
- Keep `F1_hlrange_rv10` as the best Step 9 `f1` specialist branch when the user prefers `f1` Sharpe first and is willing to accept a small `f1` return tradeoff.
- Do not treat this Step 9 result as a new repository-wide default.

## Source files

- `docs/signature/plan/step9_f1_short_results.md`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- `runs/step9_f1_full/f1_hlrange_rv10/results.csv`
- `runs/step9_f1_full/f1_hlrange_rv10/summary_by_algo_fold.csv`
- `runs/step9_f1_full/f1_hlrange_rv10/summary_by_algo.csv`
- `runs/step9_f1_full/f1_hlrange_rv10_seed42/summary_by_algo.csv`
- `runs/step9_f1_full/f1_hlrange_rv10_seed43/summary_by_algo.csv`
- `runs/step9_f1_full/f1_hlrange_rv10_seed44/summary_by_algo.csv`
- `runs/step9_f1_full/f1_hlrange_rv10_seed45/summary_by_algo.csv`
- `runs/step9_f1_full/f1_hlrange_rv10_seed46/summary_by_algo.csv`
