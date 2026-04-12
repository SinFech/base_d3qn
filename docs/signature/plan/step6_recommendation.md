# Step 6 Adoption Recommendation

_Recorded on 2026-04-11._

## Final decision

The signature exploration pass does not justify changing the repository default.

Adopt the following status table:

| Item | Status | Decision |
|---|---|---|
| `baseline` | Keep as default | `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml` remains the main D3QN reference config |
| `C4_hlrange` | Keep as exploratory branch | retain `configs/signature_step1/c4_hlrange.yaml` for robustness-oriented follow-up only |
| `C1_std` | Close | failed Step 4 short-run gate |
| `C2_bp` | Close | dropped after Step 1 screening |
| `C3_volprof` | Close | dropped after Step 1 screening |
| `C5_multi` | Close | dropped after Step 1 screening |
| `C6_deg4` | Close | dropped after Step 1 screening |
| `B1_explore` | Close | dropped after Step 1 screening |

## Why baseline stays default

The frozen ranking order from `step2.md` was:

1. `worst_fold_oos_sharpe_mean`
2. overall `oos_sharpe_mean`
3. overall `oos_return_pct_mean`

`C4_hlrange` won only the first criterion.

## Comparison summary

### Aggregate

| Config | OOS Sharpe mean | OOS Return % mean | Worst-fold OOS Sharpe mean |
|---|---:|---:|---:|
| `baseline` | `0.3132` | `45.2441` | `-0.4965` |
| `C4_hlrange` | `0.2468` | `33.3150` | `-0.3281` |
| delta (`C4_hlrange - baseline`) | `-0.0664` | `-11.9291` | `+0.1685` |

### Fold-level

| Fold | Baseline Sharpe | `C4_hlrange` Sharpe | Delta Sharpe | Baseline Return % | `C4_hlrange` Return % | Delta Return % |
|---|---:|---:|---:|---:|---:|---:|
| `f1` | `0.8905` | `0.7643` | `-0.1261` | `85.8617` | `87.6641` | `+1.8024` |
| `f2` | `0.5456` | `0.3040` | `-0.2416` | `67.4540` | `25.2615` | `-42.1925` |
| `f3` | `-0.4965` | `-0.3281` | `+0.1685` | `-17.5835` | `-12.9805` | `+4.6029` |

## Interpretation

`C4_hlrange` produced a real robustness improvement on the hardest fold:

- better `f3` Sharpe
- better `f3` return

That signal is not enough for default promotion because:

- overall OOS Sharpe regressed
- overall OOS Return regressed
- `f2` underperformed baseline by a large margin on both primary metrics

The strongest defensible reading is:

- baseline remains the better mainline config
- `C4_hlrange` remains a targeted alternative if the next question is specifically about worst-fold protection

## Repository recommendation

Use the baseline config by default in:

- future benchmark tables
- README-level D3QN references
- any comparison where one mainline D3QN config is needed

Use `C4_hlrange` only when the task is explicitly one of:

- robustness-oriented follow-up on `f3`
- investigating whether range-aware channels can reduce downside on the hardest fold
- a direct baseline-vs-`C4_hlrange` ablation under a newly defined protocol

## Closed items

The rest of the Step 1 family should stay closed for this pass:

- `C1_std`
- `C2_bp`
- `C3_volprof`
- `C5_multi`
- `C6_deg4`
- `B1_explore`

Reopening any of them should require a new note that explains why the recorded screening evidence is no longer sufficient.

## Source files

- `docs/signature/plan/step3_runtime_results.md`
- `docs/signature/plan/step4_short_results.md`
- `docs/signature/plan/step5_full_results.md`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- `runs/step5_full_c4_hlrange/summary_by_algo.csv`
- `runs/step5_full_c4_hlrange/summary_by_algo_fold.csv`
