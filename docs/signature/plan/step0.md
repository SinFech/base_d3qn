# Step 0 - Measure Baseline Metrics

## Goal

Measure and record the baseline metrics that all signature candidates will be compared against.

## Scope

This step is limited to the primary baseline defined in `baseline.md`.

It should:

- run or recover the baseline evaluation under the agreed protocol
- collect the decision metrics and diagnostic metrics already frozen in `baseline.md`
- record the exact config, run artifact paths, and summary numbers used as the reference point for later steps

It should not:

- change the signature recipe
- change the D3QN host branch settings
- compare multiple candidate upgrades
- make any adoption decision

## Inputs

- `docs/signature/plan/baseline.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `scripts/walk_forward_protocol.py`
- existing result artifacts in `runs/`, if a valid baseline run already matches the frozen definition

## Deliverables

- one confirmed baseline measurement record for the frozen signature baseline
- recorded values for:
  - `f1`
  - `f2`
  - `f3`
  - overall OOS aggregate
- for each target, recorded primary comparison metrics:
  - `oos_sharpe_mean`
  - `oos_return_pct_mean`
- and recorded diagnostic metrics when available:
  - `oos_reward_mean`
  - `oos_sharpe_std`
  - `oos_return_pct_std`
  - `is_sharpe_mean`
  - `is_sharpe_std`
- explicit references to the source files used for the measurement, such as:
  - `summary_by_algo_fold.csv`
  - `summary_by_algo.csv`
- a short note on whether the baseline numbers were reused from an existing run group or produced by a new run

## Notes

- Reuse an existing run only if it matches the frozen baseline definition closely enough to avoid mixing in unrelated config changes.
- If no existing run is clean enough, Step 0 should produce a fresh baseline measurement before Step 1 continues.
- The outcome of this step is a stable numeric reference point, not an interpretation of candidate quality.

## Evidence

- Recorded baseline measurement: `docs/signature/plan/baseline_metrics.md`
