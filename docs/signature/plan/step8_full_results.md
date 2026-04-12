# Step 8 Full Walk-Forward Results

_Recorded on 2026-04-11._

## Protocol

- Baseline control:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`
  - effective setting: `rolling_vol.window = 5`
- Candidate:
  - `RV10`
  - config: `configs/signature_step7/rv10.yaml`
- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_rolling_long_oos.json`
- Seeds:
  - `42,43,44,45,46`
- Total candidate runs:
  - `3 folds x 5 seeds = 15 runs`
- Output root:
  - `runs/step8_full_rv10/`

## Aggregate comparison

| Config | OOS Sharpe mean | OOS Return % mean | Worst-fold OOS Sharpe mean |
|---|---:|---:|---:|
| `baseline (RV5)` | `0.3132` | `45.2441` | `-0.4965` |
| `RV10` | `0.3404` | `39.6564` | `-0.3168` |
| delta (`RV10 - baseline`) | `+0.0272` | `-5.5877` | `+0.1798` |

## Per-fold comparison

| Fold | Baseline OOS Sharpe | `RV10` OOS Sharpe | Delta Sharpe | Baseline OOS Return % | `RV10` OOS Return % | Delta Return % |
|---|---:|---:|---:|---:|---:|---:|
| `f1` | `0.8905` | `0.8853` | `-0.0052` | `85.8617` | `78.0882` | `-7.7735` |
| `f2` | `0.5456` | `0.4525` | `-0.0931` | `67.4540` | `54.5574` | `-12.8966` |
| `f3` | `-0.4965` | `-0.3168` | `+0.1798` | `-17.5835` | `-13.6764` | `+3.9071` |

## Interpretation

`RV10` improved the two robustness-oriented headline targets:

- overall `oos_sharpe_mean`
- `worst_fold_oos_sharpe_mean`

The strongest improvement is still `f3`:

- better `f3` Sharpe
- better `f3` return

The tradeoff is that `RV10` gave back return elsewhere:

- `f1` return regressed while Sharpe stayed almost flat
- `f2` regressed on both primary metrics
- aggregate OOS Return regressed even though aggregate OOS Sharpe improved

This makes `RV10` more defensible than a pure short-run winner, but not clean enough to replace the default `RV5` baseline.

## Step 8 decision

- `RV10` is worth keeping as an exploratory rolling-vol alternative.
- `rolling_vol.window = 5` remains the repository default.
- `RV10` should not be promoted to default because:
  - aggregate OOS Return regressed by `5.5877` percentage points
  - `f2` degraded materially on both primary metrics
  - the Step 8 default-promotion rule required avoiding material regression on overall OOS metrics

## Recommendation

- Keep `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml` as the default reference config.
- Keep `configs/signature_step7/rv10.yaml` only for robustness-oriented follow-up where improving `f3` and worst-fold Sharpe matters more than maximizing aggregate return.

## Source files

- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- `runs/step8_full_rv10/summary_by_algo.csv`
- `runs/step8_full_rv10/summary_by_algo_fold.csv`
