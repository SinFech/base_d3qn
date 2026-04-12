# Step 8 - Run Full Walk-Forward Comparison for `RV10`

## Goal

Run the official full walk-forward comparison for the promoted `RV10` rolling-vol window candidate and determine whether it is strong enough to replace the current `window=5` default.

## Scope

This step is limited to one follow-up from Step 7:

- candidate:
  - `RV10`
- control:
  - the official `window=5` baseline

It should:

- run `RV10` under the same full walk-forward protocol used by the frozen baseline
- compare `RV10` against the official `window=5` baseline on:
  - `f1`
  - `f2`
  - `f3`
  - overall OOS aggregate
- record whether `rolling_vol.window=10` is strong enough to change the default signature recipe

It should not:

- reopen `RV3` or `RV20`
- combine the window sweep with any other signature changes
- introduce a new training budget for only one side of the comparison
- change the repository default without a recorded full result table

## Inputs

- `docs/signature/plan/baseline.md`
- `docs/signature/plan/baseline_metrics.md`
- `docs/signature/plan/step6.md`
- `docs/signature/plan/step6_recommendation.md`
- `docs/signature/plan/step7.md`
- `docs/signature/plan/step7_window_sweep_results.md`
- baseline full-run artifacts:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo.csv`
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- candidate config:
  - `configs/signature_step7/rv10.yaml`
- fold protocol:
  - `configs/folds_rolling_long_oos.json`
- runner:
  - `scripts/walk_forward_protocol.py`

## Deliverables

- one full walk-forward result record for `RV10`
- one matched comparison table against the official `window=5` baseline on:
  - `f1`
  - `f2`
  - `f3`
  - overall OOS aggregate
- one explicit keep-or-drop decision for `RV10`
- one explicit default-decision for `rolling_vol.window`

## Notes

### Candidate set

Step 7 promoted exactly one non-default window:

- `RV10`

No other rolling-vol window should be part of this full stage.

### Baseline reuse rule

The official baseline full-run remains the matched control for this step:

- run group:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`
- effective rolling-vol setting:
  - `rolling_vol.window = 5`
- protocol:
  - `configs/folds_rolling_long_oos.json`
  - seeds `42,43,44,45,46`
  - `3 folds x 5 seeds = 15 runs`

This reuse is valid because the frozen baseline recipe already is the `RV5` control.

### Frozen candidate run protocol

The new full run for `RV10` should use:

- config:
  - `configs/signature_step7/rv10.yaml`
- folds:
  - `configs/folds_rolling_long_oos.json`
- seeds:
  - `42,43,44,45,46`
- train split override:
  - `1.0`
- eval episodes:
  - `50`
- eval epsilon:
  - `0.0`
- default training budget from the baseline config family

### Comparison rule

This step is stricter than the permissive Step 7 short gate.

`RV10` is worth keeping only if the full result table shows a defensible tradeoff against the official baseline.

For a default promotion, require all of the following:

- no material regression on overall `oos_sharpe_mean`
- no material regression on overall `oos_return_pct_mean`
- no severe fold-level collapse that makes `RV10` harder to defend than `RV5`

If `RV10` improves worst-fold behavior but still loses materially on aggregate OOS, it may remain exploratory without becoming the default.

### Ranking rule

If the full results are mixed, rank the evidence in this order:

1. `worst_fold_oos_sharpe_mean`
2. overall `oos_sharpe_mean`
3. overall `oos_return_pct_mean`

This keeps the repository's robustness-first logic consistent with the earlier signature plan.

### Expected outputs

If this step is executed, write results to:

- `docs/signature/plan/step8_full_results.md`
- `runs/step8_full_rv10/`

## Evidence

- Step 7 promotion record: `docs/signature/plan/step7_window_sweep_results.md`
- Official baseline definition: `docs/signature/plan/baseline.md`
- Current default recommendation: `docs/signature/plan/step6_recommendation.md`
