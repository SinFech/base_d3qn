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

## H. 2026-04 Signature Exploration Update

This section records the dedicated signature-component sweep that branched from the frozen D3QN baseline
`configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`.

### H1. Candidate funnel

- Step 1 candidate family:
  - `C1_std`
  - `C2_bp`
  - `C3_volprof`
  - `C4_hlrange`
  - `C5_multi`
  - `C6_deg4`
  - `B1_explore`
- Shortlist after Step 1 `f3` screening:
  - `C1_std`
  - `C4_hlrange`
- Survivor after runtime and short-run gate:
  - `C4_hlrange`

### H2. Full evaluation result

Protocol:

- Runner: `scripts/walk_forward_protocol.py`
- Folds: `configs/folds_rolling_long_oos.json`
- Seeds: `42,43,44,45,46`
- Baseline control:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`
- Candidate:
  - `runs/step5_full_c4_hlrange`

Aggregate comparison:

| Config | OOS Sharpe mean | OOS Return % mean | Worst-fold OOS Sharpe mean |
|---|---:|---:|---:|
| `baseline` | `0.3132` | `45.2441` | `-0.4965` |
| `C4_hlrange` | `0.2468` | `33.3150` | `-0.3281` |
| delta (`C4_hlrange - baseline`) | `-0.0664` | `-11.9291` | `+0.1685` |

Per-fold headline deltas:

- `f1`:
  - Sharpe `-0.1261`
  - Return `+1.8024`
- `f2`:
  - Sharpe `-0.2416`
  - Return `-42.1925`
- `f3`:
  - Sharpe `+0.1685`
  - Return `+4.6029`

### H3. Interpretation

- `C4_hlrange` improved the hardest fold and therefore remains a defensible robustness-oriented branch.
- It did not earn default promotion because aggregate OOS Sharpe and OOS Return both regressed, with a large `f2` penalty.
- Repository decision:
  - keep baseline as the default D3QN signature recipe
  - keep `C4_hlrange` only as an exploratory alternative

### H4. Source files

- `docs/signature/plan/step3_runtime_results.md`
- `docs/signature/plan/step4_short_results.md`
- `docs/signature/plan/step5_full_results.md`
- `docs/signature/plan/step6_recommendation.md`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- `runs/step5_full_c4_hlrange/summary_by_algo.csv`
- `runs/step5_full_c4_hlrange/summary_by_algo_fold.csv`

## I. 2026-04 Rolling-Vol Window Sweep

This section records a narrow signature ablation that changed only the `rolling_vol.window` horizon
inside the frozen baseline recipe.

### I1. Protocol

- Config family:
  - `configs/signature_step7/rv3.yaml`
  - `configs/signature_step7/rv5.yaml`
  - `configs/signature_step7/rv10.yaml`
  - `configs/signature_step7/rv20.yaml`
- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step1_screen_f3.json`
- Fold:
  - `f3` only
- Seeds:
  - `42,43,44`
- Budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`

### I2. Short-run comparison

| Candidate | `rolling_vol.window` | OOS Sharpe mean | OOS Return % mean |
|---|---:|---:|---:|
| `RV3` | `3` | `-0.4353` | `-15.7772` |
| `RV5` | `5` | `-0.3872` | `-14.7716` |
| `RV10` | `10` | `-0.3086` | `-12.0602` |
| `RV20` | `20` | `-0.3214` | `-12.9860` |

### I3. Interpretation

- `RV10` was the strongest non-default window on both primary metrics.
- `RV20` also beat the `RV5` control, but was weaker than `RV10`.
- `RV3` lost to the control on both primary metrics.
- Repository decision after this step:
  - keep `window=5` as the default because the sweep was short-run only
  - keep `RV10` as the single promoted follow-up candidate if a later full walk-forward comparison is requested

### I4. Source files

- `docs/signature/plan/step7_window_sweep_results.md`
- `runs/step7_short_volwindow_f3/rv3/summary_by_algo.csv`
- `runs/step7_short_volwindow_f3/rv5/summary_by_algo.csv`
- `runs/step7_short_volwindow_f3/rv10/summary_by_algo.csv`
- `runs/step7_short_volwindow_f3/rv20/summary_by_algo.csv`

## J. 2026-04 `RV10` Full Walk-Forward Comparison

This section records the full follow-up on the promoted Step 7 candidate:

- `RV10`
- `rolling_vol.window = 10`

### J1. Protocol

- Candidate:
  - `configs/signature_step7/rv10.yaml`
- Baseline control:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`
  - effective rolling-vol setting: `window = 5`
- Runner:
  - `scripts/walk_forward_protocol.py`
- Folds:
  - `configs/folds_rolling_long_oos.json`
- Seeds:
  - `42,43,44,45,46`
- Output root:
  - `runs/step8_full_rv10/`

### J2. Aggregate comparison

| Config | OOS Sharpe mean | OOS Return % mean | Worst-fold OOS Sharpe mean |
|---|---:|---:|---:|
| `baseline (RV5)` | `0.3132` | `45.2441` | `-0.4965` |
| `RV10` | `0.3404` | `39.6564` | `-0.3168` |
| delta (`RV10 - baseline`) | `+0.0272` | `-5.5877` | `+0.1798` |

### J3. Per-fold comparison

| Fold | Baseline OOS Sharpe | `RV10` OOS Sharpe | Delta Sharpe | Baseline OOS Return % | `RV10` OOS Return % | Delta Return % |
|---|---:|---:|---:|---:|---:|---:|
| `f1` | `0.8905` | `0.8853` | `-0.0052` | `85.8617` | `78.0882` | `-7.7735` |
| `f2` | `0.5456` | `0.4525` | `-0.0931` | `67.4540` | `54.5574` | `-12.8966` |
| `f3` | `-0.4965` | `-0.3168` | `+0.1798` | `-17.5835` | `-13.6764` | `+3.9071` |

### J4. Interpretation

- `RV10` improved overall OOS Sharpe and materially improved the worst fold.
- `RV10` also improved both primary metrics on `f3`.
- The cost of that robustness improvement was lower return:
  - aggregate OOS Return regressed
  - `f2` regressed on both primary metrics
  - `f1` Sharpe was almost flat, but return was lower

### J5. Repository decision

- Keep `rolling_vol.window = 5` as the default.
- Keep `RV10` only as an exploratory robustness branch.

### J6. Source files

- `docs/signature/plan/step8_full_results.md`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- `runs/step8_full_rv10/summary_by_algo.csv`
- `runs/step8_full_rv10/summary_by_algo_fold.csv`

## K. 2026-04 `f1`-Only Specialist Search

This section records the Step 9 branch that intentionally ignored `f2`, `f3`, and aggregate OOS
and optimized only the `f1` out-of-sample metrics.

### K1. Short-screen protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - `42,43,44`
- Budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- Eval episodes:
  - `50`
- Batches:
  - `existing`
    - baseline plus reopened implemented candidates
  - `combos`
    - baseline plus the Step 9 combination shortlist

### K2. Short-screen results

`existing` batch:

| Candidate | `f1` OOS Sharpe mean | `f1` OOS Return % mean |
|---|---:|---:|
| `baseline` | `0.8925` | `128.8697` |
| `RV10` | `0.9131` | `84.0532` |
| `C6_deg4` | `0.9040` | `79.7812` |
| `RV20` | `0.9036` | `79.5530` |
| `C3_volprof` | `0.8710` | `66.6003` |
| `C5_multi` | `0.8486` | `97.4013` |
| `C2_bp` | `0.8442` | `96.9628` |
| `C4_hlrange` | `0.8342` | `106.8047` |
| `C1_std` | `0.8217` | `56.6999` |
| `B1_explore` | `0.6800` | `64.8236` |

`combos` batch:

| Candidate | `f1` OOS Sharpe mean | `f1` OOS Return % mean |
|---|---:|---:|
| `F1_hlrange_rv10` | `0.8957` | `91.5247` |
| `baseline` | `0.8885` | `72.2245` |
| `F1_std_rv10` | `0.8210` | `69.7954` |
| `F1_std_hlrange_rv10` | `0.8160` | `66.5855` |
| `F1_std_hlrange` | `0.7997` | `48.1886` |

Short-screen interpretation:

- No reopened singleton candidate beat the matched `existing` control strongly enough to earn promotion.
- `F1_hlrange_rv10` was the only combination candidate that improved both `f1` primary metrics against its matched `combos` baseline.
- Step 9 promotion decision:
  - promote only `F1_hlrange_rv10` into a full `f1` follow-up

### K3. Full `f1` follow-up

Candidate:

- `configs/signature_step9/f1_hlrange_rv10.yaml`

Protocol:

- fold:
  - `f1` only
- seeds:
  - `42,43,44,45,46`
- full budget family:
  - config default (`260` episodes, `100000` max total steps)
- execution detail:
  - launched as five parallel single-seed runs
  - merged into `runs/step9_f1_full/f1_hlrange_rv10/`

Full comparison versus the official baseline `f1` result:

| Config | `f1` OOS Sharpe mean | `f1` OOS Return % mean |
|---|---:|---:|
| baseline | `0.8905` | `85.8617` |
| `F1_hlrange_rv10` | `0.9072` | `84.7223` |
| delta (`candidate - baseline`) | `+0.0168` | `-1.1393` |

### K4. Interpretation

- Step 9 did not find a full-run `f1` candidate that clearly improved both primary metrics at once.
- `F1_hlrange_rv10` was still the strongest specialist discovered in this branch:
  - it improved `f1` OOS Sharpe
  - it gave up only `1.1393` return percentage points versus the official baseline
- Repository interpretation:
  - keep the official baseline as the best two-metric `f1` reference
  - keep `F1_hlrange_rv10` only as an `f1` Sharpe-tilted specialist branch

### K5. Source files

- `docs/signature/plan/step9_f1_short_results.md`
- `docs/signature/plan/step9_f1_full_results.md`
- `runs/step9_f1_short/existing/`
- `runs/step9_f1_short/combos/`
- `runs/step9_f1_full/f1_hlrange_rv10/summary_by_algo.csv`
- `runs/step9_f1_full/f1_hlrange_rv10/summary_by_algo_fold.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`

## L. 2026-04 Single-Seed Cherry-Pick Validation

This section records a narrow robustness check on the Step 9 short-run cherry-pick signals.

### L1. Promotion rule

Promote only short-run `(candidate, seed)` pairs where the same seed beat the matched Step 9 batch baseline
on both:

- `f1` OOS Sharpe
- `f1` OOS Return

Promoted set:

| Candidate | Seed | Batch |
|---|---:|---|
| `C2_bp` | `44` | `existing` |
| `F1_std_rv10` | `42` | `combos` |
| `F1_std_hlrange_rv10` | `42` | `combos` |
| `F1_hlrange_rv10` | `43` | `combos` |

Matched baseline controls:

- baseline seed `42`
- baseline seed `43`
- baseline seed `44`

### L2. Full same-seed comparison

| Candidate-seed pair | Matched baseline seed | Candidate `f1` Sharpe | Baseline `f1` Sharpe | Delta Sharpe | Candidate `f1` Return % | Baseline `f1` Return % | Delta Return % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C2_bp seed44` | `44` | `0.8799` | `0.8772` | `+0.0026` | `94.4002` | `132.5071` | `-38.1069` |
| `F1_std_rv10 seed42` | `42` | `0.8404` | `0.7003` | `+0.1400` | `56.0134` | `69.2750` | `-13.2616` |
| `F1_std_hlrange_rv10 seed42` | `42` | `0.5508` | `0.7003` | `-0.1495` | `37.3408` | `69.2750` | `-31.9342` |
| `F1_hlrange_rv10 seed43` | `43` | `0.8498` | `0.8882` | `-0.0384` | `124.9049` | `142.2286` | `-17.3237` |

### L3. Interpretation

- None of the promoted Step 9 cherry picks remained a same-seed full-run winner on both primary metrics.
- Two candidates preserved Sharpe-only upside:
  - `C2_bp seed44`
  - `F1_std_rv10 seed42`
- The other two candidates lost cleanly against their matched baseline seeds:
  - `F1_std_hlrange_rv10 seed42`
  - `F1_hlrange_rv10 seed43`

Repository interpretation:

- Step 10 is a negative robustness check on short-run single-seed cherry picks.
- Short-run cherry-pick signals in this branch were not strong enough to justify any promotion beyond exploratory note status.

### L4. Source files

- `docs/signature/plan/step10.md`
- `docs/signature/plan/step10_single_seed_full_results.md`
- `runs/step10_single_seed_f1/`

## M. 2026-04 Embedding Reduction and Replacement Cherry-Picks

This section records the two follow-up branches that focused only on `f1` cherry-pick behavior:

- Step 11: strict reductions of the baseline path embedding
- Step 12: one-for-one channel replacements within the baseline path embedding

Both branches reused the Step 9 `existing` short-run baseline control:

- `f1` OOS Sharpe mean `0.8925`
- `f1` OOS Return % mean `128.8697`

### M1. Step 11 reduction branch

Short-screen outcome:

| Candidate | Status | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Promoted seeds |
|---|---|---:|---:|---|
| `D1_price_return` | executed | `0.7258` | `71.4326` | none |
| `D2_price_vol5` | executed | `0.7711` | `93.5407` | `44` |
| `D3_return_vol5` | executed | `0.9324` | `127.2506` | `42,43` |
| `D4_price_only` | infeasible | n/a | n/a | n/a |
| `D5_return_only` | infeasible | n/a | n/a | n/a |
| `D6_vol5_only` | infeasible | n/a | n/a | n/a |

Structural note:

- `D4-D6` reduced the observation to `obs_dim = 9`
- the unchanged `ConvDuelingDQN` backbone then failed with a negative-dimension linear layer error

Full same-seed follow-up:

| Candidate-seed pair | Baseline seed | Candidate `f1` Sharpe | Baseline `f1` Sharpe | Delta Sharpe | Candidate `f1` Return % | Baseline `f1` Return % | Delta Return % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `D2_price_vol5 seed44` | `44` | `0.7556` | `0.8772` | `-0.1217` | `77.1807` | `132.5071` | `-55.3263` |
| `D3_return_vol5 seed42` | `42` | `0.8227` | `0.7003` | `+0.1224` | `108.3174` | `69.2750` | `+39.0424` |
| `D3_return_vol5 seed43` | `43` | `0.9157` | `0.8882` | `+0.0275` | `103.6459` | `142.2286` | `-38.5827` |

Reduction-branch interpretation:

- `D3_return_vol5 seed42` survived as a strict same-seed two-metric winner.
- `D3_return_vol5 seed43` preserved Sharpe-only upside.
- `D2_price_vol5 seed44` failed cleanly in the full follow-up.

### M2. Step 12 replacement branch

Short-screen outcome:

| Candidate | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Promoted seeds |
|---|---:|---:|---|
| `R1_hl_for_price` | `0.7980` | `95.6912` | none |
| `R2_hl_for_return` | `0.8577` | `90.0432` | none |
| `R3_hl_for_vol5` | `0.8791` | `115.3303` | none |
| `R4_volprof_for_price` | `0.8873` | `117.5257` | none |
| `R5_volprof_for_return` | `0.8380` | `90.6803` | `42` |
| `R6_volprof_for_vol5` | `0.8761` | `82.5308` | none |

Full same-seed follow-up:

| Candidate-seed pair | Baseline seed | Candidate `f1` Sharpe | Baseline `f1` Sharpe | Delta Sharpe | Candidate `f1` Return % | Baseline `f1` Return % | Delta Return % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `R5_volprof_for_return seed42` | `42` | `0.8358` | `0.7003` | `+0.1355` | `95.9919` | `69.2750` | `+26.7168` |

Replacement-branch interpretation:

- `R5_volprof_for_return seed42` survived as a strict same-seed two-metric winner.
- No other replacement candidate earned follow-up status.

### M3. Repository interpretation

- Step 11 and Step 12 both found narrow `f1` specialist branches that can beat the matched baseline on the right seed.
- The strongest surviving same-seed winners were:
  - `D3_return_vol5 seed42`
  - `R5_volprof_for_return seed42`
- These are still specialist notes, not repository-wide defaults.
- Default signature decisions from Steps 6 and 8 remain unchanged.

### M4. Source files

- `docs/signature/plan/step11_reduction_short_results.md`
- `docs/signature/plan/step11_reduction_single_seed_full_results.md`
- `docs/signature/plan/step12_replacement_short_results.md`
- `docs/signature/plan/step12_replacement_single_seed_full_results.md`
- `runs/step11_f1_reduction_short/`
- `runs/step11_f1_reduction_single_seed_full/`
- `runs/step12_f1_replacement_short/`
- `runs/step12_f1_replacement_single_seed_full/`

## N. 2026-04 Return-Vol Window Sweep on the `D5` Base

This section records the Step 13 follow-up that treated `D5_return_only` as a conceptual base
and asked which `rolling_vol.window` works best when paired only with `log_return`.

### N1. Family definition

Executable family:

- `log_return`
- `rolling_vol(window=k)`

Candidates:

| Candidate | Window |
|---|---:|
| `DW3_return_vol3` | `3` |
| `DW5_return_vol5` | `5` |
| `DW10_return_vol10` | `10` |
| `DW20_return_vol20` | `20` |

Structural note:

- the true one-channel `D5_return_only` embedding remained infeasible under the frozen backbone
- Step 13 therefore reused `DW5_return_vol5` as the executable family control

### N2. Short-screen protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - `42,43,44`
- Budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- Promotion rule:
  - candidate-seed must beat the matched `DW5_return_vol5` seed on both:
    - `f1` OOS Sharpe
    - `f1` OOS Return %

### N3. Short-screen result

| Candidate | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs `DW5` | Delta Return % vs `DW5` | Promoted seeds |
|---|---:|---:|---:|---:|---|
| `DW5_return_vol5` | `0.9324` | `127.2506` | `0.0000` | `0.0000` | control |
| `DW20_return_vol20` | `0.8733` | `75.1748` | `-0.0591` | `-52.0758` | none |
| `DW3_return_vol3` | `0.8183` | `99.2355` | `-0.1141` | `-28.0151` | none |
| `DW10_return_vol10` | `0.7878` | `82.6008` | `-0.1446` | `-44.6497` | none |

Closest miss:

- `DW20 seed42`
  - Sharpe beat `DW5 seed42` by `+0.0185`
  - Return still lagged by `-3.2049` percentage points

### N4. Interpretation

- No alternative window displaced `DW5_return_vol5`.
- Within this return-vol family, `window=5` remained the strongest executable horizon.
- Step 13 therefore produced no promoted candidate and no new full-run winner.

### N5. Source files

- `docs/signature/plan/step13_window_short_results.md`
- `docs/signature/plan/step13_window_single_seed_full_results.md`
- `runs/step13_f1_return_vol_window_short/`

## O. 2026-04 Replacement Reopen for the Missing `rolling_mean` Family

This section records the Step 14 follow-up that reopened the only major implemented replacement family
that had not yet entered the formal replacement protocol.

### O1. Candidate family

Baseline path embedding:

- `log_price`
- `log_return`
- `rolling_vol(window=5)`

Reopened replacement candidates:

| Candidate | Replacement interpretation |
|---|---|
| `RM1_mean_for_price` | replace `log_price` with `rolling_mean(window=5)` |
| `RM2_mean_for_return` | replace `log_return` with `rolling_mean(window=5)` |
| `RM3_mean_for_vol5` | replace `rolling_vol(window=5)` with `rolling_mean(window=5)` |

### O2. Short-screen protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - `42,43,44`
- Budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- Promotion rule:
  - candidate-seed must beat the matched baseline seed on both:
    - `f1` OOS Sharpe
    - `f1` OOS Return %

Baseline control reused from Step 9:

- mean `f1` OOS Sharpe = `0.8925`
- mean `f1` OOS Return % = `128.8697`

### O3. Short-screen result

| Candidate | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs baseline | Delta Return % vs baseline | Promoted seeds |
|---|---:|---:|---:|---:|---|
| `RM1_mean_for_price` | `0.7813` | `91.2854` | `-0.1112` | `-37.5843` | none |
| `RM2_mean_for_return` | `0.8776` | `92.5916` | `-0.0149` | `-36.2780` | none |
| `RM3_mean_for_vol5` | `0.7480` | `84.1296` | `-0.1445` | `-44.7401` | none |

Closest same-seed miss:

- `RM2_mean_for_return seed44`
  - Sharpe delta vs matched baseline seed `44`: `-0.0131`
  - Return delta vs matched baseline seed `44`: `-7.0602`

Sharpe-only spike:

- `RM2_mean_for_return seed43`
  - Sharpe delta vs matched baseline seed `43`: `+0.1277`
  - Return delta vs matched baseline seed `43`: `-51.9363`

### O4. Interpretation

- Reopening the missing `rolling_mean` family did not produce a Step 12-style replacement winner.
- No candidate passed the same-seed two-metric short gate.
- Therefore Step 14 stopped at the short stage and produced no new full-run follow-up.
- The strongest validated replacement-style specialist note remains:
  - `R5_volprof_for_return seed42`

### O5. Source files

- `docs/signature/plan/step14_replacement_reopen_short_results.md`
- `docs/signature/plan/step14_replacement_reopen_single_seed_full_results.md`
- `runs/step14_f1_replacement_reopen_short/`

## P. 2026-04 Encoder Bottleneck Validation with `MLPDuelingDQN`

This section records the Step 15 follow-up that tested whether the strongest `f1` specialist signature branches
benefit from replacing the flat-vector `ConvDuelingDQN` encoder with `MLPDuelingDQN`.

### P1. Candidate family

All Step 15 configs keep the same `f1` protocol and only change the network type to `MLPDuelingDQN`.

| Candidate | Path embedding |
|---|---|
| `MLP_baseline` | `log_price + log_return + rolling_vol(window=5)` |
| `MLP_D3_return_vol5` | `log_return + rolling_vol(window=5)` |
| `MLP_R5_volprof_for_return` | `log_price + rolling_vol(window=5) + normalized_cumulative_volume` |

### P2. Short-screen protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - `42,43,44`
- Budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- Promotion rule:
  - candidate-seed must beat the matched `MLP_baseline` seed on both:
    - `f1` OOS Sharpe
    - `f1` OOS Return %

### P3. Short-screen result

| Candidate | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs `MLP_baseline` | Delta Return % vs `MLP_baseline` | Promoted seeds |
|---|---:|---:|---:|---:|---|
| `MLP_baseline` | `0.8279` | `68.2118` | `0.0000` | `0.0000` | control |
| `MLP_D3_return_vol5` | `0.8937` | `92.2096` | `+0.0658` | `+23.9978` | `42`, `43` |
| `MLP_R5_volprof_for_return` | `0.7760` | `96.5115` | `-0.0519` | `+28.2996` | `43` |

### P4. Full `f1` comparison

Official repository reference:

- baseline `f1` `5`-seed average:
  - `Sharpe 0.8905`
  - `Return 85.8617%`

Full MLP results:

| Config | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs official baseline | Delta Return % vs official baseline |
|---|---:|---:|---:|---:|
| `MLP_baseline` | `0.7437` | `77.2559` | `-0.1468` | `-8.6058` |
| `MLP_D3_return_vol5` | `0.8073` | `77.9949` | `-0.0832` | `-7.8667` |
| `MLP_R5_volprof_for_return` | `0.8059` | `100.1231` | `-0.0846` | `+14.2614` |

Important internal comparison versus `MLP_baseline`:

- `MLP_D3_return_vol5`:
  - `Sharpe +0.0636`
  - `Return +0.7391 pct`
- `MLP_R5_volprof_for_return`:
  - `Sharpe +0.0622`
  - `Return +22.8672 pct`

### P5. Interpretation

- The encoder choice clearly interacts with signature design:
  - both specialist branches improved versus the matched MLP control
- But a blanket swap to `MLPDuelingDQN` is not a repository-level fix:
  - the baseline embedding regressed materially under MLP
  - no MLP candidate beat the official baseline on both full-run `f1` metrics
- The strongest Step 15 branch was:
  - `MLP_R5_volprof_for_return`
  - it improved return strongly
  - it remained below the official baseline on Sharpe

### P6. Source files

- `docs/signature/plan/step15_mlp_short_results.md`
- `docs/signature/plan/step15_mlp_full_results.md`
- `runs/step15_f1_mlp_short/`
- `runs/step15_f1_mlp_full/mlp_baseline/`
- `runs/step15_f1_mlp_full/mlp_d3_return_vol5/`
- `runs/step15_f1_mlp_full/mlp_r5_volprof_for_return/`

## Q. 2026-04 Step 11 Supplement: Full `5`-Seed Mean for `D3_return_vol5`

This section backfills the missing full `5`-seed `f1` mean for the strongest Step 11 reduction branch.

### Q1. Candidate

| Candidate | Path embedding |
|---|---|
| `D3_return_vol5` | `log_return + rolling_vol(window=5)` |

### Q2. Protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - `42,43,44,45,46`
- Budget:
  - config default
  - `train.num_episodes = 260`
  - `train.max_total_steps = 100000`
- Eval episodes:
  - `50`

### Q3. Result

Official baseline reference:

- `Sharpe 0.8905`
- `Return 85.8617%`

`D3_return_vol5` full `5`-seed mean:

- `Sharpe 0.8728`
- `Return 95.3811%`

Delta versus official baseline:

- `Sharpe -0.0176`
- `Return +9.5194 pct`

### Q4. Interpretation

- `D3_return_vol5` is stronger than a fragile same-seed-only cherry-pick.
- The config preserves an `f1` return advantage at the full-mean level.
- It still does not become a strict two-metric winner because the full-mean Sharpe remains slightly below the official baseline.

### Q5. Source files

- `docs/signature/plan/step11_reduction_full_mean_results.md`
- `runs/step11_f1_reduction_full_mean/d3_return_vol5/summary_by_algo_fold.csv`
- `runs/step11_f1_reduction_full_mean/d3_return_vol5/summary_by_algo.csv`

## R. 2026-04 Step 16 `logsig.degree` Sweep

This section records the focused `f1`-only sweep over the signature truncation level while keeping the
baseline embedding and `ConvDuelingDQN` encoder fixed.

### R1. Candidate family

| Candidate | `logsig.degree` | Status |
|---|---:|---|
| `L1_deg1` | `1` | structurally infeasible |
| `L2_deg2` | `2` | executed short screen |
| `L3_deg3` | `3` | executed short screen, family control |
| `L4_deg4` | `4` | executed short screen |

### R2. Protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - `42,43,44`
- Budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- Promotion rule:
  - candidate-seed must beat the matched `L3_deg3` seed on both:
    - `f1` OOS Sharpe
    - `f1` OOS Return %

### R3. Structural note

- `L1_deg1` reduced the observation size to `8`.
- Under the frozen `ConvDuelingDQN` backbone, that produced a negative hidden dimension and could not be run.

### R4. Short-screen result

| Candidate | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs `L3_deg3` | Delta Return % vs `L3_deg3` | Promoted seeds |
|---|---:|---:|---:|---:|---|
| `L2_deg2` | `0.7435` | `94.9931` | `-0.1724` | `-5.5530` | none |
| `L3_deg3` | `0.9159` | `100.5461` | `0.0000` | `0.0000` | control |
| `L4_deg4` | `0.8841` | `96.9654` | `-0.0319` | `-3.5807` | none |

Closest partial wins:

- `L2_deg2 seed42`:
  - Return improved strongly
  - Sharpe still missed the `L3_deg3` control
- `L4_deg4 seed43`:
  - Sharpe improved
  - Return regressed
- `L4_deg4 seed44`:
  - Return improved
  - Sharpe regressed

### R5. Interpretation

- `degree=3` remains the strongest balanced short-run setting in the frozen baseline family.
- No alternative degree produced a same-seed two-metric short-run win over `degree=3`.
- No candidate entered the Step 16 full stage, so this step produced no new official-baseline cherry-pick winner.

### R6. Source files

- `docs/signature/plan/step16.md`
- `docs/signature/plan/step16_degree_short_results.md`
- `docs/signature/plan/step16_degree_single_seed_full_results.md`
- `runs/step16_f1_degree_short/l2_deg2/`
- `runs/step16_f1_degree_short/l3_deg3/`
- `runs/step16_f1_degree_short/l4_deg4/`
