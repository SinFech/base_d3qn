# Step 9 - Run an `f1`-Only Specialist Search

## Goal

Search for a signature configuration that improves the `f1` out-of-sample targets only:

- `f1 oos_sharpe_mean`
- `f1 oos_return_pct_mean`

This step intentionally ignores:

- `f2`
- `f3`
- overall OOS aggregate

## Scope

This step is a targeted branch, not a replacement for the earlier multi-fold plan.

It should:

- define a compact `f1`-focused shortlist extracted from prior steps
- run the first screen on `f1`, not `f3`
- rank candidates only by the two `f1` OOS primary metrics
- identify zero, one, or multiple `f1`-specialist candidates worth keeping

It should not:

- revise the repository-wide default decision from Steps 6 and 8
- treat `f1`-only results as a replacement for the multi-fold baseline
- require candidates to be good on `f2`, `f3`, or overall OOS
- widen scope beyond signature-only changes

## Inputs

- `docs/signature/plan/baseline.md`
- `docs/signature/plan/baseline_metrics.md`
- `docs/signature/plan/step1_screening_f3_results.md`
- `docs/signature/plan/step5_full_results.md`
- `docs/signature/plan/step7_window_sweep_results.md`
- `docs/signature/plan/step8_full_results.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/signature_step1/`
- `configs/signature_step7/`
- `configs/folds_rolling_long_oos.json`
- `scripts/walk_forward_protocol.py`

## Deliverables

- one frozen `f1`-focused candidate shortlist
- one `f1`-only short-screen protocol
- one `f1`-only full-evaluation protocol
- one explicit `f1` ranking rule
- one explicit note that Step 9 produces an `f1` specialist branch, not a new global default

## Notes

### Target baseline for this step

This step still uses the same baseline config family:

- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`

But the only comparison target that matters is the official `f1` baseline:

- `f1 oos_sharpe_mean = 0.8904662600702572`
- `f1 oos_return_pct_mean = 85.86166745250878`

### Why Step 9 may reopen previously filtered ideas

Earlier steps used `f3` and worst-fold robustness as the main filter.
Step 9 intentionally changes the objective.

That means a candidate may be worth revisiting here even if it was not strong enough for the earlier robustness-first path, as long as prior evidence gives it a plausible `f1`-specific reason.

The practical rule for Step 9 is:

- earlier `f3`-based screening is informative
- but it is not exclusionary

So Step 9 must not inherit the Step 1 / Step 4 shortlist as a hard filter.

### Reopened Step 9 candidate universe

Step 9 should reopen all already-implemented signature candidates that were previously screened under non-`f1` objectives.

That reopened universe is:

| Family | Candidate ID | Source | Step 9 status |
|---|---|---|---|
| Control | `baseline` | frozen baseline | required |
| Step 1 single | `C1_std` | `configs/signature_step1/c1_std.yaml` | keep |
| Step 1 single | `C2_bp` | `configs/signature_step1/c2_bp.yaml` | keep |
| Step 1 single | `C3_volprof` | `configs/signature_step1/c3_volprof.yaml` | keep |
| Step 1 single | `C4_hlrange` | `configs/signature_step1/c4_hlrange.yaml` | keep |
| Step 1 single | `C5_multi` | `configs/signature_step1/c5_multi.yaml` | keep |
| Step 1 single | `C6_deg4` | `configs/signature_step1/c6_deg4.yaml` | keep |
| Step 1 bundle | `B1_explore` | `configs/signature_step1/b1_explore.yaml` | keep |
| Step 7 window | `RV10` | `configs/signature_step7/rv10.yaml` | keep |
| Step 7 window | `RV20` | `configs/signature_step7/rv20.yaml` | keep |

`RV3` stays out by default because it already lost to the matched `RV5` control even in the short window sweep and does not have a separate `f1`-specific positive signal.

### Frozen `f1` combination shortlist

In addition to the reopened implemented configs, Step 9 should explicitly test a small combination set built only from previously validated primitives.

| Tier | Candidate ID | Construction | Reason to keep for `f1` |
|---|---|---|---|
| Pair | `F1_std_hlrange` | `C1_std + C4_hlrange` | combine scale normalization with the only prior `f1` return-positive channel |
| Pair | `F1_std_rv10` | `C1_std + RV10` | combine a cheap normalization change with the strongest rolling-vol follow-up |
| Pair | `F1_hlrange_rv10` | `C4_hlrange + RV10` | combine the strongest prior `f1` return-positive channel with the strongest rolling-vol follow-up |
| Triple | `F1_std_hlrange_rv10` | `C1_std + C4_hlrange + RV10` | most direct compact combination of the Step 9 `f1`-plausible primitives |

### Config rule

Every Step 9 candidate should be derived from prior validated primitives only.

The allowed primitives are:

- baseline recipe
- `standardize_path_channels = true`
- `basepoint = true`
- `normalized_cumulative_volume`
- `high_low_range`
- `subwindow_sizes = [8, 16, 24]`
- `logsig.degree = 4`
- `rolling_vol.window = 10`
- `rolling_vol.window = 20`

No new signature primitive should be introduced in this step.

### Recommended execution protocol

#### `S9_short_f1`

Use one cheap `f1`-only screen first.

Because the reopened universe is now larger, Step 9 may split the short screen into two matched batches:

- `Batch A_existing`
  - baseline plus all reopened implemented configs
- `Batch B_combos`
  - baseline plus the Step 9 combination shortlist

If the short screen is split this way, each batch must include its own matched baseline control under the same budget.

Common protocol:

- fold protocol:
  - one single-fold JSON that keeps only `f1`
- seeds:
  - `42,43,44`
- default short budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- eval episodes:
  - `50`

#### `S9_full_f1`

Promote only the strongest short-run survivors into a full `f1`-only evaluation:

- fold protocol:
  - one single-fold JSON that keeps only `f1`
- seeds:
  - `42,43,44,45,46`
- full budget:
  - same full training budget family as the baseline config

Do not automatically promote every short-run winner.

Prefer promoting:

- the strongest singleton candidate
- the strongest combination candidate

unless the short-run results are nearly tied and a third candidate is needed to resolve the ranking.

### Ranking rule

Because the user goal is to improve both `f1` primary metrics, rank candidates in this order:

1. `f1 oos_sharpe_mean`
2. `f1 oos_return_pct_mean`

If a later note wants return-first ranking, it must say so explicitly.

### Pass rule

For the short screen, a candidate is worth promoting only if:

- it beats the `f1` baseline on at least one primary metric:
  - higher `f1 oos_sharpe_mean`, or
  - higher `f1 oos_return_pct_mean`
- and it does not regress catastrophically on the other:
  - no worse than `0.05` Sharpe points, or
  - no worse than `5.0` return-percentage points

This pass rule is intentionally local to `f1` only.

There is no Step 9 penalty for being weak on:

- `f2`
- `f3`
- overall OOS

### Step 9 outcome type

Even if Step 9 finds a strong `f1` winner, that result should be interpreted as:

- an `f1` specialist branch

not:

- a new repository-wide default

That broader default decision still belongs to the earlier multi-fold steps unless the plan is explicitly widened again.

### Expected outputs

If Step 9 is executed, write results to:

- `docs/signature/plan/step9_f1_short_results.md`
- optionally `docs/signature/plan/step9_f1_full_results.md`
- `runs/step9_f1_short/`
- optionally `runs/step9_f1_full/`

## Evidence

- Official baseline source: `docs/signature/plan/baseline.md`
- `f1`-positive signal from `C4_hlrange`: `docs/signature/plan/step5_full_results.md`
- rolling-vol follow-up evidence: `docs/signature/plan/step8_full_results.md`
- initial candidate-family history: `docs/signature/plan/step1_screening_f3_results.md`
