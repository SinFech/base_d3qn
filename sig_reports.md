# Signature `f1` Specialist Report

## Scope

This report summarizes the current `f1`-focused signature candidates against the official repository baseline.

Reference baseline:

- config:
  - `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- official `f1` `5`-seed average:
  - `OOS Sharpe = 0.8905`
  - `OOS Return % = 85.8617`
- source:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`

Cherry-pick rule used in this report:

- choose one full-run seed that beats the official baseline on both:
  - `f1 OOS Sharpe`
  - `f1 OOS Return %`
- if multiple seeds satisfy that rule, keep the best one by:
  1. higher `Sharpe`
  2. then higher `Return`

Two rows use a slightly broader cherry-pick note for completeness:

- `baseline` reports the best retained full-run baseline seed among the preserved controls
- `MLP_R5_volprof_for_return` reports the best same-seed win versus the matched `MLP_baseline`
  because it has no two-metric winner against the official baseline

## Summary Table

| Config | Change vs baseline | Cherry-pick note | Full `5`-seed mean | Read |
|---|---|---|---|---|
| `baseline` | reference config | `seed43`: `Sharpe 0.8882`, `Return 142.2286%` | `Sharpe 0.8905`, `Return 85.8617%` | official repository `f1` baseline reference |
| `C4_hlrange` | add `high_low_range` | `seed46`: `Sharpe 0.9494`, `Return 120.9950%` | `Sharpe 0.7643`, `Return 87.6641%` | strong single-seed upside, full mean keeps Return only |
| `F1_hlrange_rv10` | add `high_low_range`; change `rolling_vol.window 5 -> 10` | `seed44`: `Sharpe 1.0010`, `Return 87.4066%` | `Sharpe 0.9072`, `Return 84.7223%` | best Sharpe-first specialist |
| `D3_return_vol5` | remove `log_price`; keep `log_return + rolling_vol(5)` | `seed43`: `Sharpe 0.9157`, `Return 103.6459%` | `Sharpe 0.8728`, `Return 95.3811%` | strongest Conv-based `f1` specialist branch so far |
| `MLP_D3_return_vol5` | same embedding as `D3_return_vol5`, but switch encoder `ConvDuelingDQN -> MLPDuelingDQN` | `seed45`: `Sharpe 0.9303`, `Return 114.6148%` | `Sharpe 0.8073`, `Return 77.9949%` | single-seed upside exists, but full mean does not hold |
| `MLP_R5_volprof_for_return` | replace `log_return` with `normalized_cumulative_volume`; switch encoder to `MLPDuelingDQN` | `seed44`: `Sharpe 0.8905`, `Return 85.7583%` | `Sharpe 0.8059`, `Return 100.1231%` | strongest Return-first MLP branch, but no two-metric cherry-pick vs official baseline |

## Candidate Notes

### `baseline`

- path embedding:
  - `log_price`
  - `log_return`
  - `rolling_vol(window=5)`

### `C4_hlrange`

- config:
  - `configs/signature_step1/c4_hlrange.yaml`
- path embedding:
  - `log_price`
  - `log_return`
  - `rolling_vol(window=5)`
  - `high_low_range`
- best cherry-pick source:
  - `runs/step5_full_c4_hlrange/results.csv`
- full-mean source:
  - `runs/step5_full_c4_hlrange/summary_by_algo_fold.csv`

### `F1_hlrange_rv10`

- config:
  - `configs/signature_step9/f1_hlrange_rv10.yaml`
- path embedding:
  - `log_price`
  - `log_return`
  - `rolling_vol(window=10)`
  - `high_low_range`
- best cherry-pick source:
  - `runs/step9_f1_full/f1_hlrange_rv10/results.csv`
- full-mean source:
  - `runs/step9_f1_full/f1_hlrange_rv10/summary_by_algo_fold.csv`

### `D3_return_vol5`

- config:
  - `configs/signature_step11/d3_return_vol5.yaml`
- path embedding:
  - `log_return`
  - `rolling_vol(window=5)`
- removed channel:
  - `log_price`
- best cherry-pick source:
  - `runs/step11_f1_reduction_single_seed_full/d3_return_vol5_seed43/summary_by_algo.csv`
- full-mean source:
  - `runs/step11_f1_reduction_full_mean/d3_return_vol5/summary_by_algo_fold.csv`

### `MLP_D3_return_vol5`

- config:
  - `configs/signature_step15/mlp_d3_return_vol5.yaml`
- path embedding:
  - `log_return`
  - `rolling_vol(window=5)`
- encoder change:
  - `model.type: conv_dueling -> mlp_dueling`
- best cherry-pick source:
  - `runs/step15_f1_mlp_full/mlp_d3_return_vol5/results.csv`
- full-mean source:
  - `runs/step15_f1_mlp_full/mlp_d3_return_vol5/summary_by_algo_fold.csv`

### `MLP_R5_volprof_for_return`

- config:
  - `configs/signature_step15/mlp_r5_volprof_for_return.yaml`
- path embedding:
  - `log_price`
  - `rolling_vol(window=5)`
  - `normalized_cumulative_volume`
- replaced channel:
  - `log_return`
- encoder change:
  - `model.type: conv_dueling -> mlp_dueling`
- best matched-MLP-baseline cherry-pick source:
  - `runs/step15_f1_mlp_full/mlp_r5_volprof_for_return/results.csv`
  - `runs/step15_f1_mlp_full/mlp_baseline/results.csv`
- full-mean source:
  - `runs/step15_f1_mlp_full/mlp_r5_volprof_for_return/summary_by_algo_fold.csv`

## Important Additional Note

`R5_volprof_for_return` under the original Conv encoder remains an important strict same-seed winner:

- config:
  - `configs/signature_step12/r5_volprof_for_return.yaml`
- change:
  - replace `log_return` with `normalized_cumulative_volume`
- same-seed full win:
  - baseline `seed42`: `Sharpe 0.7003`, `Return 69.2750%`
  - `R5_volprof_for_return seed42`: `Sharpe 0.8358`, `Return 95.9919%`
- source:
  - `docs/signature/plan/step12_replacement_single_seed_full_results.md`

This config is not included in the main table above because it still does not have a full `5`-seed mean.

## Main Takeaways

- The strongest balanced specialist branch so far is:
  - `D3_return_vol5`
  - it gives up only a small amount of Sharpe
  - while adding a meaningful full-mean Return gain
- The strongest Sharpe-first branch is:
  - `F1_hlrange_rv10`
- The strongest Return-first branch is:
  - `MLP_R5_volprof_for_return`
