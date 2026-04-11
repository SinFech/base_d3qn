# Baseline Metrics Record

_Recorded on 2026-04-11 for Step 0._

## Decision

Step 0 uses the retained walk-forward run group below as the plan baseline:

- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`

This supersedes the earlier temporary idea of using the newer `sell_fractions` reference branch as the planning baseline.

## Why this run group is the baseline

- The user explicitly chose the strong `f1` result from this run group as the baseline reference point.
- `reports/experiments.md` already records both the per-fold and aggregate values for this run group.
- The retained artifact set includes both:
  - `summary_by_algo_fold.csv`
  - `summary_by_algo.csv`
- Those files are sufficient for the comparison targets required by the current plan:
  - `f1`
  - `f2`
  - `f3`
  - overall OOS aggregate

## Source files used

Primary metric sources:

- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo.csv`

Supporting provenance sources:

- `reports/experiments.md`
- `reports/progress.md`
- `configs/folds_rolling_long_oos.json`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `scripts/walk_forward_protocol.py`

## Baseline reference table

### Fold-level targets

| Target | `oos_sharpe_mean` | `oos_return_pct_mean` | `oos_reward_mean` |
|---|---:|---:|---:|
| `f1` | `0.8904662600702572` | `85.86166745250878` | `-1.11878036601586` |
| `f2` | `0.5455914259601037` | `67.45400861226639` | `-1.1786028248457279` |
| `f3` | `-0.49653040461622266` | `-17.583455920137908` | `-0.989829244246644` |

### Aggregate OOS target

| Target | `oos_sharpe_mean` | `oos_return_pct_mean` | `oos_reward_mean` | `oos_sharpe_std` | `oos_return_pct_std` | `is_sharpe_mean` | `is_sharpe_std` | `worst_fold_id` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| overall OOS | `0.3131757604713794` | `45.24407338154575` | `-1.0957374783694105` | `0.6320700614788746` | `54.61919699764876` | `0.9420942173775809` | `0.16529451167602918` | `f3` |

### Headline baseline chosen by the user

The user-selected headline baseline is the `f1` row:

- `oos_sharpe_mean = 0.8904662600702572`
- `oos_return_pct_mean = 85.86166745250878`

## Comparison note

Later signature experiments should still report all four targets:

- `f1`
- `f2`
- `f3`
- overall OOS

The current screening rule is:

- if any one of those targets beats the corresponding baseline target on `oos_sharpe_mean` or `oos_return_pct_mean`, the candidate is worth keeping for further consideration

## Reuse note

No new training run was launched for Step 0.

The Step 0 baseline numbers were recovered from an existing retained run group because the necessary fold-level and aggregate summary tables are already present and match the values reported in the repository reports.
