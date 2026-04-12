# Step 12 Replacement Single-Seed Full-Run Results

## Scope

This note records the executed `S12_single_seed_full` follow-up from Step 12.

Only one replacement candidate-seed pair was promoted from the Step 12 short screen:

- `R5_volprof_for_return seed42`

Shared matched baseline control reused from Step 10:

- `baseline seed42`

## Protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - one seed per run
- Full budget family:
  - config default
  - effective default in this family:
    - `train.num_episodes = 260`
    - `train.max_total_steps = 100000`
- Eval episodes:
  - `50`

## Result

`R5_volprof_for_return` uses:

- `log_price`
- `rolling_vol(window=5)`
- `normalized_cumulative_volume`

and replaces:

- `log_return`

| Config | `f1` OOS Sharpe | `f1` OOS Return % |
|---|---:|---:|
| baseline seed `42` | `0.7003` | `69.2750` |
| `R5_volprof_for_return` seed `42` | `0.8358` | `95.9919` |
| delta (`candidate - baseline`) | `+0.1355` | `+26.7168` |

## Interpretation

- The Step 12 short-run cherry-pick signal survived cleanly under full training.
- `R5_volprof_for_return seed42` remained a strict same-seed two-metric winner versus the matched baseline control.
- This makes `R5_volprof_for_return seed42` the strongest replacement-style `f1` cherry-pick found in Step 12.

## Step 12 Decision

Repository interpretation:

- Keep `R5_volprof_for_return seed42` as a valid replacement-style `f1` specialist note.
- Do not promote it to a repository-wide default based on a single-seed specialist result.
- No other Step 12 replacement candidate earned follow-up status.

## Source files

- `docs/signature/plan/step12.md`
- `docs/signature/plan/step12_replacement_short_results.md`
- `docs/signature/plan/step10_single_seed_full_results.md`
- `runs/step10_single_seed_f1/baseline_seed42/summary_by_algo.csv`
- `runs/step12_f1_replacement_single_seed_full/r5_volprof_for_return_seed42/summary_by_algo.csv`
