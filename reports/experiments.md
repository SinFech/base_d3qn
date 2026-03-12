# Experiment Results (Current Curated Set)

This report only includes active runs currently kept in `runs/`.
It is intended to be a clean, reproducible reference for current model comparisons.

## Protocol

- Dataset: `data/Bitcoin History 2010-2024.csv`
- Train/IS range: `2014-01-01 ~ 2020-12-31`
- OOS range: `2021-01-01 ~ 2024-02-09`
- Eval episodes: `50`
- Eval settings: `epsilon=0.0`, `fixed_windows=true`, `seed=20240101`
- Primary comparison metrics:
  - `sharpe_ratio`
  - `mean_return_rate_pct`
  - `mean_reward_return`

## A. Cross-Algorithm Comparison (Capital-Aware)

| Run | Algo | IS Sharpe | OOS Sharpe | IS Return % | OOS Return % | IS Reward | OOS Reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ppo_cash_best_oos_e200_s42_1c4d96` | PPO | 1.1491 | 0.6046 | 13.6233 | 2.7714 | 0.0013 | 0.0045 |
| `ppo_cash_tuned_conservative_e200_s42_3df81c` | PPO | 0.8889 | -0.3400 | 73.2551 | -12.0485 | 0.0123 | 0.0197 |
| `sac_cash_baseline_e200_s42_c4ef82` | SAC | 0.6955 | -0.3844 | 61.8171 | -13.3729 | -0.0070 | 0.0158 |
| `sac_cash_tuned_low_entropy_e200_s42_a848d1` | SAC | 0.7299 | -0.3528 | 116.3273 | -12.0744 | -0.0238 | -0.0239 |

Interpretation:
- PPO currently provides the strongest OOS performance in the curated set.
- SAC tuning improved IS metrics but did not turn OOS Sharpe positive.

## B. D3QN Cash-Constrained Comparison

| Run | Setting Focus | IS Sharpe | OOS Sharpe | IS Return % | OOS Return % | IS Reward | OOS Reward |
|---|---|---:|---:|---:|---:|---:|---:|
| `d3qn_cash_grid_best_oos_a06_n3_e50_s42_98c991` | Risk-grid best OOS checkpoint (50 ep) | 0.5760 | -0.2479 | 200.3411 | -10.8321 | 0.0103 | 0.0079 |
| `d3qn_cash_6act_exposure08_sellall_e200_s42_8864ec` | 6-action, exposure 0.8 | 0.8600 | -0.3346 | 42.5904 | -10.0104 | 0.0156 | -0.0069 |
| `d3qn_ablation_base_seed43_e200_716920` | Ablation baseline (no PER/n-step) | 0.7809 | -0.6077 | 200.2523 | -18.9298 | -0.0698 | -0.0067 |
| `d3qn_ablation_per_n3_seed43_e200_b1f6dc` | Ablation PER + n-step | 0.8044 | -0.4622 | 210.1763 | -19.8298 | -0.0454 | -0.0296 |

Interpretation:
- In this snapshot, D3QN OOS remains below PPO and mostly negative on Sharpe.
- `PER + n-step` improved IS stability in ablation but did not yet deliver positive OOS Sharpe.
- Exposure `0.8` was materially better than `0.6` for OOS Sharpe in 6-action sell-all tests.

## C. Legacy Reference (No-Cash Constraint)

| Run | Context | IS Sharpe | OOS Sharpe | IS Return % | OOS Return % |
|---|---|---:|---:|---:|---:|
| `d3qn_legacy_nocash_sr_enhanced_sellall_pos5_e200_df84f3` | Legacy env reference | 0.6103 | -0.0988 | 1314.7931 | -27.3364 |

Interpretation:
- This run is retained only as a historical baseline to show how no-cash constraints inflate IS return-scale.

## D. Curated Run Registry

Active historical single-run directories retained in `runs/`:
- `d3qn_ablation_base_seed43_e200_716920`
- `d3qn_ablation_per_n3_seed43_e200_b1f6dc`
- `d3qn_cash_6act_exposure08_sellall_e200_s42_8864ec`
- `d3qn_cash_grid_best_oos_a06_n3_e50_s42_98c991`
- `d3qn_legacy_nocash_sr_enhanced_sellall_pos5_e200_df84f3`
- `ppo_cash_best_oos_e200_s42_1c4d96`
- `ppo_cash_tuned_conservative_e200_s42_3df81c`
- `sac_cash_baseline_e200_s42_c4ef82`
- `sac_cash_tuned_low_entropy_e200_s42_a848d1`

## E. Practical Conclusion

- Current best OOS benchmark in this repository: `ppo_cash_best_oos_e200_s42_1c4d96`.
- D3QN remains competitive in IS but needs further OOS robustness work (multi-seed and risk-control tuning).
- Future comparison should use multi-seed confidence intervals rather than single-seed point estimates.

## F. 2026-03 Rolling OOS Update (New)

This section appends the latest walk-forward multi-seed results while keeping the historical curated snapshot above.

### F1. OOS Training Loop Setup

- Runner: `scripts/walk_forward_protocol.py`
- Folds source: `configs/folds_rolling_long_oos.json`
- Seeds: `42,43,44,45,46` (5 seeds per fold)
- Total runs:
  - `runs/batch_wf_rolling_long_oos_repeat`: `ppo + d3qn`, `3 folds x 5 seeds = 30 runs`
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`: `d3qn only`, `3 folds x 5 seeds = 15 runs`
- Common evaluation settings (from resolved configs):
  - `eval.num_episodes=50`
  - `eval.epsilon=0.0`
  - `eval.fixed_windows=true`
  - `eval.fixed_windows_seed=20240101`
- Loop semantics in `scripts/walk_forward_protocol.py`:
  - For each `(algo, fold, seed)`: set train range to fold train dates and train once.
  - Evaluate IS on the same fold train range.
  - Evaluate OOS on the fold test range.
  - Write per-run summaries and aggregate into `results.csv`, `summary_by_algo_fold.csv`, `summary_by_algo.csv`.
- Fold windows used:
  - `f1`: train `2014-01-01 ~ 2018-12-31`, test `2019-01-01 ~ 2022-12-31`
  - `f2`: train `2015-01-01 ~ 2019-12-31`, test `2020-01-01 ~ 2023-12-31`
  - `f3`: train `2016-01-01 ~ 2020-12-31`, test `2021-01-01 ~ 2024-02-09`

### F2. Aggregate OOS Comparison (Mean Across 15 Runs)

| Run group | Algo/config | OOS Sharpe (mean +/- std) | OOS Return % (mean +/- std) | Worst fold OOS Sharpe |
|---|---|---:|---:|---:|
| `runs/batch_wf_rolling_long_oos_repeat` | PPO baseline | `0.4705 +/- 0.5948` | `32.84 +/- 43.36` | `-0.1884 (f3)` |
| `runs/batch_wf_rolling_long_oos_repeat` | D3QN baseline (3-action, no PER, n=1) | `0.0945 +/- 0.8288` | `26.41 +/- 48.69` | `-0.9421 (f3)` |
| `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold` | D3QN (6-action, PER=0.6, n=3) | `0.3132 +/- 0.6321` | `45.24 +/- 54.62` | `-0.4965 (f3)` |

### F3. Per-Fold Means (5 Seeds per Fold)

| Algo/config | Fold | OOS Sharpe mean | OOS Return % mean |
|---|---|---:|---:|
| PPO baseline | `f1` | `1.1079` | `66.35` |
| PPO baseline | `f2` | `0.4920` | `41.46` |
| PPO baseline | `f3` | `-0.1884` | `-9.28` |
| D3QN baseline (3-action, no PER, n=1) | `f1` | `0.8650` | `68.60` |
| D3QN baseline (3-action, no PER, n=1) | `f2` | `0.3607` | `36.98` |
| D3QN baseline (3-action, no PER, n=1) | `f3` | `-0.9421` | `-26.35` |
| D3QN (6-action, PER=0.6, n=3) | `f1` | `0.8905` | `85.86` |
| D3QN (6-action, PER=0.6, n=3) | `f2` | `0.5456` | `67.45` |
| D3QN (6-action, PER=0.6, n=3) | `f3` | `-0.4965` | `-17.58` |

### F4. Source Files

- `runs/batch_wf_rolling_long_oos_repeat/summary_by_algo.csv`
- `runs/batch_wf_rolling_long_oos_repeat/summary_by_algo_fold.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`

## G. 2026-03 Action-Space Comparison (6 vs 7 vs 8 Actions)

This section keeps the latest D3QN action-space study while retaining the older milestone runs above.

### G1. Protocol

- Runner: `scripts/walk_forward_protocol.py`
- Folds: `configs/folds_rolling_long_oos.json`
- Seeds: `42,43,44,45,46`
- Common loop:
  - rolling walk-forward
  - `3 folds x 5 seeds = 15 runs`
  - `train_split=1.0`
  - `train_episodes=260`
  - later reruns used `max_total_steps=130000` to avoid premature stopping around episode `210`
- Primary decision metric:
  - `worst_fold_oos_sharpe_mean`
  - then `oos_sharpe_mean`
  - then `oos_return_pct_mean`

### G2. Aggregate Comparison

| Run group | D3QN config | OOS Sharpe (mean +/- std) | OOS Return % (mean +/- std) | Worst fold OOS Sharpe |
|---|---|---:|---:|---:|
| `runs/wf_d3qn_sellfrac_wf_sellfrac_20260308_125429` | `6-action`, sell fractions | `0.2920 +/- 0.4391` | `15.24 +/- 21.12` | `-0.1985 (f3)` |
| `runs/wf_d3qn_7act_sellfrac_wf_7act_sellfrac_20260309_011013` | `7-action`, old buy ladder + sell fractions | `0.2536 +/- 0.5370` | `21.04 +/- 34.79` | `-0.3711 (f3)` |
| `runs/wf_d3qn_8act_sellfrac_wf_8act_sellfrac_20260309_122333` | `8-action`, add `buy 100%` | `0.2764 +/- 0.5332` | `30.58 +/- 39.61` | `-0.3919 (f3)` |

### G3. Per-Fold Comparison

| Config | Fold | OOS Sharpe mean | OOS Return % mean |
|---|---|---:|---:|
| `6-action` | `f1` | `0.7302` | `34.63` |
| `6-action` | `f2` | `0.3444` | `17.14` |
| `6-action` | `f3` | `-0.1985` | `-6.05` |
| `7-action` | `f1` | `0.8533` | `61.17` |
| `7-action` | `f2` | `0.2788` | `15.23` |
| `7-action` | `f3` | `-0.3711` | `-13.28` |
| `8-action` | `f1` | `0.8306` | `68.46` |
| `8-action` | `f2` | `0.3906` | `39.95` |
| `8-action` | `f3` | `-0.3919` | `-16.69` |

### G4. Interpretation

- `6-action sell_fractions` remains the best D3QN mainline on robustness.
- `7-action` and `8-action` both improve upside in `f1/f2`, but they do so by becoming more aggressive.
- `f3` deteriorates materially once the action space is enlarged:
  - `6-action f3 Sharpe = -0.1985`
  - `7-action f3 Sharpe = -0.3711`
  - `8-action f3 Sharpe = -0.3919`
- Current evidence says larger action spaces are not adding structural edge; they are adding upside beta and downside fragility.

### G5. Current D3QN Recommendation

- Keep `6-action sell_fractions` as the current D3QN branch to compare against PPO.
- Treat `7-action` and `8-action` as informative negative controls: useful for understanding aggressiveness, but not the primary deployment candidate.
- If D3QN iteration continues, prioritize risk-budget sweeps on the `6-action` branch instead of further action expansion.
