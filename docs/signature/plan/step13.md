# Step 13 - Sweep `rolling_vol.window` on Top of the `D5` Return-Only Base

## Goal

Test whether the strongest executable continuation of the `D5_return_only` idea is a
`log_return + rolling_vol(window=k)` family rather than the fixed `window=5` branch used in Step 11.

This step stays inside the current `f1` specialist scope:

- only `f1`
- only the two primary `f1` metrics
- allow cherry-pick interpretation on the candidate side
- keep the official baseline reference fixed at the repository baseline `f1` `5`-seed average

## Scope

This step should:

- treat `D5_return_only` as the conceptual base:
  - keep `log_return`
  - do not bring `log_price` back
- vary only the `rolling_vol.window` horizon added on top of that return-only base
- keep the path embedding size fixed at two channels:
  - `log_return`
  - one `rolling_vol(window=k)` channel
- keep the non-embedding parts of the frozen baseline recipe unchanged
- remain fully inside the Step 9 to Step 12 `f1`-specialist branch

This step should not:

- reopen additive fourth-channel experiments
- reopen replacement channels such as `high_low_range` or `normalized_cumulative_volume`
- reinterpret `f2`, `f3`, or aggregate OOS behavior
- change the repository-wide default configuration

## Why This Step Exists

Step 11 established:

- `D5_return_only` itself is structurally infeasible under the frozen backbone
- `D3_return_vol5` was the strongest executable reduction branch

Step 7 and Step 8 established:

- `rolling_vol.window` can matter materially
- `window=10` was the strongest non-default baseline-family horizon

So the natural next question is:

- if `log_return` is the base signal, which `rolling_vol.window` is best when paired with it?

## Inputs

- `docs/signature/plan/step7.md`
- `docs/signature/plan/step8.md`
- `docs/signature/plan/step11.md`
- `docs/signature/plan/step11_reduction_short_results.md`
- `docs/signature/plan/step11_reduction_single_seed_full_results.md`
- `docs/signature/plan/baseline_metrics.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/folds_signature_step9_f1.json`
- `scripts/walk_forward_protocol.py`

## Baseline and Control Rules

Official external reference for final reporting:

- baseline `f1` `5`-seed average from:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`

This official baseline stays fixed at:

- `f1 oos_sharpe_mean = 0.8904662600702572`
- `f1 oos_return_pct_mean = 85.86166745250878`

Local family control for the sweep:

- `DW5_return_vol5`

Reason:

- it is the direct executable continuation of `D5_return_only`
- it is identical in path embedding to `D3_return_vol5`
- it provides the correct within-family comparison for the new window sweep

## Structural Note

`D5_return_only` itself should not be run again in this step.

Reason:

- Step 11 already showed that the one-channel `log_return` embedding collapses to `obs_dim = 9`
- the frozen `ConvDuelingDQN` backbone then fails with a negative-dimension linear layer

Therefore Step 13 uses `D5_return_only` only as a conceptual base, not as an executable control.

## Frozen Candidate Set

All Step 13 candidates must use exactly two path channels:

- `log_return`
- `rolling_vol(window=k)`

| Candidate ID | Path embedding | Role |
|---|---|---|
| `DW3_return_vol3` | `log_return + rolling_vol(window=3)` | shorter-horizon test |
| `DW5_return_vol5` | `log_return + rolling_vol(window=5)` | family control, same embedding as `D3_return_vol5` |
| `DW10_return_vol10` | `log_return + rolling_vol(window=10)` | medium-horizon test |
| `DW20_return_vol20` | `log_return + rolling_vol(window=20)` | long-horizon test |

Not included:

- `window <= 1`
- `window > 24`
- any extra path channel beyond the two listed above

## Config Rule

Every Step 13 config must:

- start from the frozen baseline config family
- keep the same account features
- keep the same signature transform settings
- keep the same D3QN backbone
- change only:
  - remove `log_price`
  - keep `log_return`
  - set `embedding.rolling_vol.window = k`

## Protocol

### `S13_short_f1_window`

Run one short `f1`-only sweep:

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

Batch composition:

- `DW3_return_vol3`
- `DW5_return_vol5`
- `DW10_return_vol10`
- `DW20_return_vol20`

Short-screen promotion rule:

- promote only candidate-seed pairs where the same short-run seed beats the matched `DW5_return_vol5` seed on both:
  - `f1 oos_sharpe`
  - `f1 oos_return_pct`

### `S13_single_seed_full`

For every promoted candidate-seed pair, run:

- one single-seed full `f1` evaluation for the promoted candidate
- one same-seed full `DW5_return_vol5` control if that exact seed is not already available

Full budget family:

- config default
- effective default in this family:
  - `train.num_episodes = 260`
  - `train.max_total_steps = 100000`

## Final Reporting Rule

Step 13 should report winners using this final interpretation:

1. candidate full-run seed must beat the official baseline `f1` `5`-seed average on both:
   - `OOS Sharpe`
   - `OOS Return %`
2. if one config has multiple winning seeds, keep only the best cherry-pick seed
3. choose that best seed in this order:
   - higher `f1 OOS Sharpe`
   - then higher `f1 OOS Return %`

## Deliverables

- one config family:
  - `configs/signature_step13/`
- one short-run result note:
  - `docs/signature/plan/step13_window_short_results.md`
- one optional full result note:
  - `docs/signature/plan/step13_window_single_seed_full_results.md`
- one short-run output root:
  - `runs/step13_f1_return_vol_window_short/`
- one optional full-run output root:
  - `runs/step13_f1_return_vol_window_full/`

## Outcome Type

Even if Step 13 finds a strong cherry-pick winner, the interpretation remains:

- an `f1` specialist branch built on the `D5` return-only idea

not:

- a repository-wide default decision
