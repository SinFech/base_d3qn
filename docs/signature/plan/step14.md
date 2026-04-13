# Step 14 - Reopen Replacement Candidates Previously Filtered by Non-Signal Gates

## Goal

Freeze one replacement-only `f1` specialist step that reopens embedding candidates which were
previously excluded for reasons other than a clean "this replacement has no value" conclusion.

This step intentionally excludes:

- embedding reduction
- additive extra-channel bundles
- pure rolling-vol horizon sweeps

It focuses only on replacement-style candidates.

## Scope

This step should:

- stay inside the `f1` specialist line introduced in Steps 9 to 13
- keep the final external reference fixed at the repository baseline `f1` `5`-seed average
- allow candidate-side cherry-pick interpretation in the final reporting layer
- reopen only replacement-style embedding candidates
- keep new code limited to generic feature-side additions when needed

This step should not:

- reopen reduction candidates such as `D4`, `D5`, or `D6`
- reopen additive families such as `C4_hlrange` or `F1_hlrange_rv10`
- reinterpret `f2`, `f3`, or aggregate OOS as the primary target
- add candidate-specific hacks that only rescue one config
- treat a single-seed `f1` specialist win as a repository-wide default decision

## Primary Metrics

All execution and reporting in this step should stay centered on:

- `f1 oos_sharpe`
- `f1 oos_return_pct`

## External Reference

The fixed final reference remains the official baseline `f1` `5`-seed average from:

- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`

Reference values:

- `f1 oos_sharpe_mean = 0.8904662600702572`
- `f1 oos_return_pct_mean = 85.86166745250878`

## Why This Step Exists

The earlier steps left one obvious replacement gap:

- `rolling_mean` is already implemented in `PathBuilder`
- it never entered the formal replacement sweep in Step 12
- so it was filtered by scope, not by performance evidence

By contrast:

- `high_low_range` replacement candidates were already tested in Step 12
- `normalized_cumulative_volume` replacement candidates were already tested in Step 12
- rolling-vol horizon substitutions were already studied in Steps 7 and 8

So the next clean replacement-only step is not to reopen everything.
It is:

- reopen the missing `rolling_mean` replacement family

## Inputs

- `docs/signature/plan/step1.md`
- `docs/signature/plan/step12.md`
- `docs/signature/plan/step12_replacement_short_results.md`
- `docs/signature/plan/step12_replacement_single_seed_full_results.md`
- `docs/signature/plan/step13.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/folds_signature_step9_f1.json`
- `rl/features/signature.py`
- `scripts/walk_forward_protocol.py`

## Replacement Baseline

Current baseline path embedding:

- `log_price`
- `log_return`
- `rolling_vol(window=5)`

Current strongest validated replacement-style specialist note:

- `R5_volprof_for_return seed42`

Interpretation rule:

- baseline remains the mandatory control for promotion
- `R5_volprof_for_return` is an incumbent replacement reference, not the promotion gate

## Frozen Candidate Set

### Family A - Implemented but previously excluded replacement channels

These channels are already supported by `PathBuilder`, but they never entered the replacement sweep.

The missing family is `rolling_mean`.

| Candidate ID | Path embedding | Replacement interpretation | Previous block reason |
|---|---|---|---|
| `RM1_mean_for_price` | `log_return + rolling_vol(window=5) + rolling_mean(window=5)` | replace `log_price` | not admitted into Step 12 scope |
| `RM2_mean_for_return` | `log_price + rolling_vol(window=5) + rolling_mean(window=5)` | replace `log_return` | not admitted into Step 12 scope |
| `RM3_mean_for_vol5` | `log_price + log_return + rolling_mean(window=5)` | replace `rolling_vol(window=5)` | not admitted into Step 12 scope |

### Already answered and therefore not reopened

These replacement families already have a clean answer from earlier steps:

- `R1_hl_for_price`
- `R2_hl_for_return`
- `R3_hl_for_vol5`
- `R4_volprof_for_price`
- `R5_volprof_for_return`
- `R6_volprof_for_vol5`

Reason:

- they were already executed under the proper replacement protocol in Step 12
- `R5_volprof_for_return` already survived as a valid same-seed full-run specialist note

### Deferred even with new code allowed

The following ideas remain out of scope for this step even though new code is now acceptable:

- OHLC intrabar micro-path
- partial lead-lag path variants
- inventory or account-history path channels

Reason:

- they are not clean one-for-one replacements of the current baseline channels
- they deserve their own future step instead of being mixed into the missing `rolling_mean` replacement sweep

## Code Rule

Step 14 may add new code only if needed to support a generic replacement channel family.

For the current frozen candidate set, the expected path is:

- no backbone change
- no environment change
- only config materialization unless an implementation gap is discovered

If an implementation gap appears, any code change must be:

- generic
- shared across replacement candidates
- safe for baseline under the same code path

## Protocol

### `S14_short_f1_replace_reopen`

Run one short `f1` replacement screen with a shared baseline control:

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

- baseline
- `RM1_mean_for_price`
- `RM2_mean_for_return`
- `RM3_mean_for_vol5`

Short promotion rule:

- promote only candidate-seed pairs where the same short-run seed beats the matched baseline seed on both:
  - `f1 oos_sharpe`
  - `f1 oos_return_pct`

### `S14_single_seed_full`

For every promoted pair:

- run one full `f1` single-seed follow-up for the candidate
- run one same-seed full baseline control if that seed is not already available

Full budget family:

- config default
- effective default in this family:
  - `train.num_episodes = 260`
  - `train.max_total_steps = 100000`

## Final Reporting Rule

The final reporting layer should keep one cherry-pick seed per config.

Reporting rule:

1. candidate full-run seed must beat the official baseline `f1` `5`-seed average on both:
   - `OOS Sharpe`
   - `OOS Return %`
2. if one config has multiple winning seeds, keep only the best one
3. choose the best seed in this order:
   - higher `f1 OOS Sharpe`
   - then higher `f1 OOS Return %`

## Deliverables

- one new planning note:
  - `docs/signature/plan/step14.md`
- one implementation/config family:
  - `configs/signature_step14/`
- one short-screen result note:
  - `docs/signature/plan/step14_replacement_reopen_short_results.md`
- one optional single-seed full result note:
  - `docs/signature/plan/step14_replacement_reopen_single_seed_full_results.md`
- one short-run output root:
  - `runs/step14_f1_replacement_reopen_short/`
- one optional full-run output root:
  - `runs/step14_f1_replacement_reopen_single_seed_full/`

## Outcome Type

Even if Step 14 finds a strong winner, the interpretation remains:

- an `f1` specialist replacement branch

not:

- a repository-wide default change
