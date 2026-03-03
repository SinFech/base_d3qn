# OOS Summary (Key Results Only)

Last updated: `2026-03-02`

## Evaluation protocol

- Dataset: `data/Bitcoin History 2010-2024.csv`
- OOS range: `2021-01-01` to `2024-02-09`
- Method: rolling walk-forward, `3 folds x 5 seeds = 15 runs` per algorithm config
- Main decision metric: `worst_fold_oos_sharpe` (robustness first), then OOS Sharpe mean and OOS Return %

## Key OOS results

| Run group | Algo/config | OOS Sharpe (mean +/- std) | OOS Return % (mean +/- std) | Worst fold (Sharpe) |
|---|---|---:|---:|---:|
| `runs/batch_wf_rolling_long_oos_repeat` | PPO baseline | `0.4705 +/- 0.5948` | `32.84 +/- 43.36` | `f3: -0.1884` |
| `runs/batch_wf_rolling_long_oos_repeat` | D3QN baseline (3-action, no PER, n=1) | `0.0945 +/- 0.8288` | `26.41 +/- 48.69` | `f3: -0.9421` |
| `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold` | D3QN (6-action, PER=0.6, n=3) | `0.3132 +/- 0.6321` | `45.24 +/- 54.62` | `f3: -0.4965` |

## Decision

- Deployment baseline remains PPO because worst-fold OOS Sharpe is strongest (`-0.1884`).
- New D3QN (`6-action, PER=0.6, n=3`) improves mean OOS Sharpe/Return versus D3QN baseline.
- D3QN still needs `f3` downside improvement before replacing PPO as primary model.

## Source summary files

- `runs/batch_wf_rolling_long_oos_repeat/summary_by_algo.csv`
- `runs/batch_wf_rolling_long_oos_repeat/summary_by_algo_fold.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
