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
| `d3qn_cash_6act_exposure06_sellall_e200_s42_9d1d1e` | 6-action, exposure 0.6 | 0.8671 | -1.1378 | 28.7134 | -20.6099 | 0.0113 | -0.0132 |
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

Active run directories retained in `runs/`:
- `d3qn_ablation_base_seed43_e200_716920`
- `d3qn_ablation_per_n3_seed43_e200_b1f6dc`
- `d3qn_cash_6act_exposure06_sellall_e200_s42_9d1d1e`
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
