# Step 5 Full Walk-Forward Results

_Recorded on 2026-04-11._

## Protocol

- Baseline control:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`
- Candidate:
  - `C4_hlrange`
  - config: `configs/signature_step1/c4_hlrange.yaml`
- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_rolling_long_oos.json`
- Seeds:
  - `42,43,44,45,46`
- Total candidate runs:
  - `3 folds x 5 seeds = 15 runs`
- Output root:
  - `runs/step5_full_c4_hlrange/`

## Aggregate comparison

| Config | OOS Sharpe mean | OOS Return % mean | Worst-fold OOS Sharpe mean |
|---|---:|---:|---:|
| `baseline` | `0.3132` | `45.2441` | `-0.4965` |
| `C4_hlrange` | `0.2468` | `33.3150` | `-0.3281` |
| delta (`C4_hlrange - baseline`) | `-0.0664` | `-11.9291` | `+0.1685` |

## Per-fold comparison

| Fold | Baseline OOS Sharpe | `C4_hlrange` OOS Sharpe | Delta Sharpe | Baseline OOS Return % | `C4_hlrange` OOS Return % | Delta Return % |
|---|---:|---:|---:|---:|---:|---:|
| `f1` | `0.8905` | `0.7643` | `-0.1261` | `85.8617` | `87.6641` | `+1.8024` |
| `f2` | `0.5456` | `0.3040` | `-0.2416` | `67.4540` | `25.2615` | `-42.1925` |
| `f3` | `-0.4965` | `-0.3281` | `+0.1685` | `-17.5835` | `-12.9805` | `+4.6029` |

## Interpretation

- `C4_hlrange` materially improved the hardest fold:
  - better `f3` Sharpe
  - better `f3` return
- `C4_hlrange` also slightly improved `f1` return, but lost `f1` Sharpe.
- The major tradeoff is `f2`:
  - `C4_hlrange` underperformed baseline sharply on both primary metrics there.
- At the aggregate level:
  - overall `oos_sharpe_mean` regressed
  - overall `oos_return_pct_mean` regressed
  - worst-fold robustness improved

## Step 5 keep-rule decision

Under the permissive keep rule from `baseline.md`, `C4_hlrange` is worth keeping for Step 6 discussion because:

- it beats the official baseline on at least one comparison target
- specifically:
  - `f1` on `oos_return_pct_mean`
  - `f3` on both primary metrics

## Step 5 decision summary

- `C4_hlrange` should remain in scope for Step 6.
- It is not a clean replacement for baseline.
- Its value is as a robustness-oriented alternative:
  - weaker aggregate upside
  - better worst-fold behavior

## Source files

- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- `runs/step5_full_c4_hlrange/summary_by_algo.csv`
- `runs/step5_full_c4_hlrange/summary_by_algo_fold.csv`
