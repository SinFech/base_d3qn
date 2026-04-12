# Step 10 - Validate Short-Run Single-Seed Cherry Picks with Full `f1` Runs

## Goal

Test whether the strongest Step 9 short-run single-seed cherry-pick signals survive under full `f1` training.

This step intentionally focuses on:

- one candidate
- one seed
- one fold

per evaluation unit.

## Scope

This step should:

- extract candidate-seed pairs from Step 9 short runs
- require that the same short-run seed beat its matched batch baseline on both primary `f1` metrics
- run full `f1` single-seed follow-ups for those promoted candidate-seed pairs
- run matched single-seed baseline controls for the same seed IDs
- write a direct candidate-vs-baseline conclusion for each promoted seed

This step should not:

- replace the Step 9 multi-seed conclusion
- reinterpret a single-seed win as a new default decision
- widen scope beyond `f1` and beyond already-implemented signature configs

## Inputs

- `docs/signature/plan/step9.md`
- `docs/signature/plan/step9_f1_short_results.md`
- `docs/signature/plan/step9_f1_full_results.md`
- `docs/signature/plan/baseline_metrics.md`
- `configs/folds_signature_step9_f1.json`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/signature_step1/c2_bp.yaml`
- `configs/signature_step9/f1_std_rv10.yaml`
- `configs/signature_step9/f1_hlrange_rv10.yaml`
- `configs/signature_step9/f1_std_hlrange_rv10.yaml`
- `scripts/walk_forward_protocol.py`

## Promotion Rule

Promote a Step 9 short-run candidate-seed pair only if:

- it is a single concrete `(candidate, seed)` pair
- that same seed beats the matched Step 9 batch baseline on both:
  - `f1 oos_sharpe`
  - `f1 oos_return_pct`

Under that rule, the promoted set is:

| Candidate | Config | Seed | Batch |
|---|---|---:|---|
| `C2_bp` | `configs/signature_step1/c2_bp.yaml` | `44` | `existing` |
| `F1_std_rv10` | `configs/signature_step9/f1_std_rv10.yaml` | `42` | `combos` |
| `F1_std_hlrange_rv10` | `configs/signature_step9/f1_std_hlrange_rv10.yaml` | `42` | `combos` |
| `F1_hlrange_rv10` | `configs/signature_step9/f1_hlrange_rv10.yaml` | `43` | `combos` |

Matched baseline controls required for this step:

- baseline seed `42`
- baseline seed `43`
- baseline seed `44`

## Protocol

Run each promoted candidate-seed pair as a single-seed full `f1` evaluation:

- runner:
  - `scripts/walk_forward_protocol.py`
- fold protocol:
  - `configs/folds_signature_step9_f1.json`
- fold:
  - `f1` only
- seeds:
  - exactly one seed per run
- full budget family:
  - config default
  - effective baseline family default:
    - `train.num_episodes = 260`
    - `train.max_total_steps = 100000`
- eval episodes:
  - `50`

Matched controls:

- run the baseline config for every seed that appears in the promoted set
- compare each candidate only to the same-seed baseline full result

## Ranking Rule

This step is diagnostic, not a new funnel.

Within the promoted set, summarize in this order:

1. same-seed delta `f1 oos_sharpe`
2. same-seed delta `f1 oos_return_pct`

## Deliverables

- one executed candidate-seed shortlist
- one executed same-seed baseline control set
- one result note:
  - `docs/signature/plan/step10_single_seed_full_results.md`
- one experiment root:
  - `runs/step10_single_seed_f1/`

## Outcome Type

Even if a candidate wins on a single seed full run, that result means:

- the cherry-pick signal survived on that specific seed

not:

- the candidate is superior in the multi-seed sense

The multi-seed Step 9 conclusion remains the main decision record.
