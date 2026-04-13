# Progress Log

This log tracks implementation milestones that materially changed model behavior, evaluation quality, or reproducibility.

## 2026-02-21
- Summary: Introduced PPO training/evaluation on the same market data pipeline used by D3QN.
- Changes:
  - Added `rl/algos/ppo/` and CLI scripts `scripts/train_ppo.py`, `scripts/eval_ppo.py`.
  - Added configurable PPO training loop (GAE, clipped policy objective, minibatch updates, checkpoints).
- Blockers: OOS robustness uncertain from single-seed runs.
- Next: Add continuous-action accounting and compare with D3QN.

## 2026-02-22
- Summary: Added continuous capital-aware environment and SAC baseline.
- Changes:
  - Added explicit ledger-style accounting (`cash`, `position`, `equity`) with costs/slippage in continuous env flow.
  - Added `rl/algos/sac/` and CLI scripts `scripts/train_sac.py`, `scripts/eval_sac.py`.
  - Added shared evaluation protocol for IS/OOS comparisons.
- Blockers: OOS return-rate remained unstable across early runs.
- Next: Improve action/risk controls and tune policies.

## 2026-02-23
- Summary: Expanded discrete capital environment action controls.
- Changes:
  - Added fractional action support via `buy_fractions` and `sell_fractions`.
  - Added exposure-cap workflow with `max_exposure_ratio` for discrete capital runs.
  - Completed exposure/action-granularity sweeps for D3QN.
- Blockers: Larger action spaces (10/17 actions) degraded OOS in single-seed tests.
- Next: Improve D3QN sample efficiency/stability before further action expansion.

## 2026-02-24
- Summary: Upgraded D3QN with PER + n-step and added guardrails for n-step truncation.
- Changes:
  - Added `PrioritizedReplayBuffer` and weighted Huber loss path.
  - Added n-step transition buffering and configurable `n_step`.
  - Fixed time-limit truncation handling so non-terminal truncation is not forced terminal in n-step flush.
  - Added config fields to `configs/default.yaml` and `configs/d3qn_signature_capital.yaml`:
    - `per_enabled`, `per_alpha`, `per_beta_start`, `per_beta_steps`, `per_eps`, `n_step`.
  - Added tests in `tests/d3qn/test_nstep_per_behavior.py`.
- Blockers: Need robust OOS comparison under fixed protocol.
- Next: Re-run D3QN/PPO/SAC under the same IS/OOS ranges.

## 2026-02-25
- Summary: Completed comparative benchmarking runs and tuned baselines.
- Changes:
  - Finalized PPO and SAC tuned runs and evaluated on OOS `2021-01-01 ~ 2024-02-09`.
  - Finalized D3QN ablation/control runs for exposure and PER+n-step comparisons.
- Blockers: Most D3QN/SAC variants remain negative OOS Sharpe; PPO best run is currently strongest OOS.
- Next: Use PPO best policy as OOS benchmark and iterate D3QN risk controls.

## 2026-02-26
- Summary: Curated run artifacts and refreshed documentation.
- Changes:
  - Cleaned `runs/` and kept only key comparison runs used by current reporting.
  - Renamed retained runs to human-readable, protocol-aware names.
  - Rewrote `README.md` and experiment report to match current repository state.
- Blockers: None.
- Next: Multi-seed evaluation for top 2-3 configs and confidence-interval reporting.

## 2026-03-02
- Summary: Finished rolling long-OOS 15-run D3QN stress test (6-action, PER=0.6, n-step=3) and compared it against PPO/D3QN baselines under the same protocol.
- Changes:
  - Completed `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold` (3 folds x 5 seeds).
  - Recorded final OOS summary:
    - D3QN (6act/PER/n3): OOS Sharpe mean `0.3132`, OOS Return mean `45.2441%`, worst-fold Sharpe `-0.4965`.
    - D3QN baseline (3-action, n=1, no PER): OOS Sharpe mean `0.0945`, worst-fold Sharpe `-0.9421`.
    - PPO baseline: OOS Sharpe mean `0.4705`, worst-fold Sharpe `-0.1884`.
  - Consolidated run-directory naming with `single_*` / `batch_*` prefixes and moved obsolete batches into `runs/_pruned_20260302/`.
- Blockers:
  - D3QN improved mean OOS metrics but still underperforms PPO on worst-fold (`f3`) robustness.
- Next:
  - Keep PPO as deployment baseline.
  - If continuing D3QN iteration, prioritize reducing `f3` downside before optimizing average return.

## 2026-03-09
- Summary: Completed D3QN action-space comparison under the same rolling walk-forward protocol and tightened repository retention rules around run artifacts.
- Changes:
  - Finished `6-action`, `7-action`, and `8-action` D3QN comparisons with `3 folds x 5 seeds`.
  - Added and retained key configs for:
    - `6-action sell_fractions`
    - `7-action old-buy + sell_fractions`
    - `8-action add buy100 + sell_fractions`
    - paired `6 vs 8` risk-budget comparison configs
  - Updated repository policy so `runs/` keeps only lightweight summary tables while checkpoints, tensorboard logs, per-run eval outputs, and resolved configs are ignored.
  - Updated README and experiment report to preserve both historical milestones and the latest action-space comparison.
- Key result:
  - `6-action sell_fractions` remains the strongest D3QN branch on `worst_fold_oos_sharpe_mean`.
  - `7-action` and `8-action` improved `f1/f2` upside but materially worsened `f3`, indicating more aggressiveness rather than better generalization.
- Next:
  - Use `6-action sell_fractions` as the D3QN reference branch.
  - Compare `6-action` vs `8-action` only under matched risk budgets if action-space expansion is revisited.

## 2026-04-11
- Summary: Completed the signature-component exploration pass against the frozen D3QN baseline and recorded the default-config decision.
- Changes:
  - Finished Step 1 candidate freeze and created the `configs/signature_step1/` config family.
  - Completed `f3` screening, runtime checks, short-run gate, and full walk-forward evaluation for the surviving signature candidate.
  - Added `scripts/benchmark_signature_wrapper.py` to benchmark the real `SignatureObsWrapper -> get_state()` observation path.
  - Recorded the final signature recommendation in `docs/signature/plan/step6_recommendation.md`.
- Key result:
  - `C4_hlrange` improved hardest-fold robustness (`f3`) but regressed aggregate OOS Sharpe and OOS Return versus the baseline.
  - Baseline remains the default D3QN signature recipe.
- Next:
  - Keep `C4_hlrange` only as a robustness-oriented alternative branch if signature work continues.
  - Do not reopen other Step 1 signature candidates without a new scope note.

## 2026-04-11
- Summary: Completed a focused `rolling_vol.window` short sweep around the frozen baseline signature recipe.
- Changes:
  - Added the `configs/signature_step7/` family for `RV3`, `RV5`, `RV10`, and `RV20`.
  - Ran the matched `f3` short-run comparison with `3` seeds under the same budget used by the earlier short-stage gate.
  - Recorded the sweep outcome in `docs/signature/plan/step7_window_sweep_results.md`.
- Key result:
  - `RV10` (`rolling_vol.window = 10`) was the strongest non-default window on both primary short-run metrics.
  - `RV20` also improved over `RV5`, but ranked below `RV10`.
  - `RV3` underperformed the `RV5` control.
- Next:
  - Keep `RV10` as the single promoted follow-up if a later full official comparison is requested.
  - Keep `window=5` as the default until such a full comparison is actually recorded.

## 2026-04-11
- Summary: Completed the full walk-forward evaluation for `RV10` against the official `rolling_vol.window=5` baseline.
- Changes:
  - Ran `RV10` over `f1/f2/f3` with `5` seeds per fold and merged the sub-fold outputs into `runs/step8_full_rv10/`.
  - Recorded the final comparison in `docs/signature/plan/step8_full_results.md`.
- Key result:
  - `RV10` improved overall OOS Sharpe (`0.3132 -> 0.3404`) and worst-fold Sharpe (`-0.4965 -> -0.3168`).
  - `RV10` regressed overall OOS Return (`45.2441% -> 39.6564%`) and underperformed baseline on `f2`.
  - The default rolling-vol window remains `5`.
- Next:
  - Keep `RV10` only as a robustness-oriented alternative.
  - Do not change the shipped default unless a later step prefers the robustness tradeoff over aggregate return.

## 2026-04-12
- Summary: Completed the Step 9 `f1`-only specialist search and recorded the best `f1` follow-up branch.
- Changes:
  - Reopened the full implemented signature-candidate universe under an `f1`-only short-screen protocol.
  - Added the `configs/signature_step9/` combination configs and ran the `existing` and `combos` short-screen batches under `configs/folds_signature_step9_f1.json`.
  - Promoted only `F1_hlrange_rv10` from the short screen and completed its full `5`-seed `f1` follow-up.
  - Merged the parallel seed-level full outputs into `runs/step9_f1_full/f1_hlrange_rv10/`.
- Key result:
  - No candidate improved both full-run `f1` primary metrics at the same time.
  - `F1_hlrange_rv10` improved `f1` OOS Sharpe (`0.8905 -> 0.9072`) but slightly regressed `f1` OOS Return (`85.8617% -> 84.7223%`).
- Next:
  - Keep the official baseline as the best two-metric `f1` reference.
  - Keep `F1_hlrange_rv10` only as an `f1` Sharpe-tilted specialist branch if that narrower objective is revisited.

## 2026-04-12
- Summary: Validated the strongest Step 9 short-run single-seed cherry picks with matched single-seed full `f1` runs.
- Changes:
  - Added `docs/signature/plan/step10.md` and executed four promoted `(candidate, seed)` follow-ups plus matched baseline seeds.
  - Recorded the same-seed full-run comparison in `docs/signature/plan/step10_single_seed_full_results.md`.
  - Kept all Step 10 outputs under `runs/step10_single_seed_f1/`.
- Key result:
  - No promoted cherry-pick candidate remained a same-seed two-metric winner under full training.
  - `C2_bp seed44` and `F1_std_rv10 seed42` preserved Sharpe gains only, while `F1_hlrange_rv10 seed43` and `F1_std_hlrange_rv10 seed42` lost on both metrics versus their matched baseline seeds.
- Next:
  - Treat Step 10 as a negative robustness check on short-run cherry-pick signals.
  - Do not promote any Step 10 candidate beyond exploratory note status.

## 2026-04-12
- Summary: Completed the embedding-reduction and embedding-replacement `f1` specialist branches and identified two same-seed full-run winners.
- Changes:
  - Added `configs/signature_step11/` and `configs/signature_step12/` for strict reduction and one-for-one replacement families.
  - Ran the Step 11 and Step 12 `f1` short screens against the shared Step 9 short baseline control.
  - Recorded the short results in:
    - `docs/signature/plan/step11_reduction_short_results.md`
    - `docs/signature/plan/step12_replacement_short_results.md`
  - Completed the promoted single-seed full follow-ups and recorded the results in:
    - `docs/signature/plan/step11_reduction_single_seed_full_results.md`
    - `docs/signature/plan/step12_replacement_single_seed_full_results.md`
- Key result:
  - Step 11 reduction branch:
    - single-channel reductions were structurally infeasible under the frozen baseline backbone
    - `D3_return_vol5 seed42` survived as a strict full-run two-metric winner
    - `D3_return_vol5 seed43` preserved Sharpe only
    - `D2_price_vol5 seed44` failed under full follow-up
  - Step 12 replacement branch:
    - `R5_volprof_for_return seed42` survived as a strict full-run two-metric winner
    - no other replacement candidate earned follow-up status
- Next:
  - Keep `D3_return_vol5 seed42` and `R5_volprof_for_return seed42` only as `f1` specialist notes.
  - Do not reinterpret these same-seed wins as repository-wide default changes without a new multi-seed scope.

## 2026-04-12
- Summary: Completed the Step 13 `D5`-based return-vol window sweep and found no reason to move away from `window=5` in that family.
- Changes:
  - Added the `configs/signature_step13/` family for:
    - `DW3_return_vol3`
    - `DW5_return_vol5`
    - `DW10_return_vol10`
    - `DW20_return_vol20`
  - Reused Step 11 `D3_return_vol5` as the `DW5_return_vol5` control because the embedding and protocol were identical.
  - Ran the new Step 13 short sweep for `DW3`, `DW10`, and `DW20`.
  - Recorded the result in:
    - `docs/signature/plan/step13_window_short_results.md`
    - `docs/signature/plan/step13_window_single_seed_full_results.md`
- Key result:
  - No alternative window beat `DW5_return_vol5` on both same-seed `f1` metrics.
  - `DW20 seed42` came closest but still missed on return.
  - No Step 13 candidate was promoted into a full follow-up.
- Next:
  - Keep `window=5` as the strongest executable horizon inside the `log_return + rolling_vol(window=k)` family.
  - If this line is revisited again, it should only be under a new scope note rather than another same-family resweep.

## 2026-04-12
- Summary: Reopened the missing replacement-only `rolling_mean` family and closed it without any promoted `f1` specialist winner.
- Changes:
  - Added the `configs/signature_step14/` family for:
    - `RM1_mean_for_price`
    - `RM2_mean_for_return`
    - `RM3_mean_for_vol5`
  - Reused the shared Step 9 short baseline control because the `f1` short protocol was identical.
  - Ran the Step 14 replacement reopen short screen and recorded the result in:
    - `docs/signature/plan/step14_replacement_reopen_short_results.md`
    - `docs/signature/plan/step14_replacement_reopen_single_seed_full_results.md`
- Key result:
  - No `rolling_mean` replacement candidate beat the matched baseline seed on both primary `f1` metrics.
  - `RM2_mean_for_return` was the closest mean-level challenger on Sharpe, but it still lost badly on return.
  - No Step 14 candidate was promoted into a full follow-up.
- Next:
  - Keep `R5_volprof_for_return seed42` as the strongest validated replacement-style specialist note.
  - If replacement work continues, open a new scope beyond the now-tested `high_low_range`, `normalized_cumulative_volume`, and `rolling_mean` families.

## 2026-04-12
- Summary: Validated the encoder-bottleneck hypothesis with an `MLPDuelingDQN` follow-up on the strongest `f1` specialist signature branches.
- Changes:
  - Added the `configs/signature_step15/` family for:
    - `mlp_baseline`
    - `mlp_d3_return_vol5`
    - `mlp_r5_volprof_for_return`
  - Ran the Step 15 `f1` short screen under the MLP encoder and recorded the result in:
    - `docs/signature/plan/step15_mlp_short_results.md`
  - Promoted both `MLP_D3_return_vol5` and `MLP_R5_volprof_for_return` into full `5`-seed `f1` follow-ups.
  - Recorded the final comparison in:
    - `docs/signature/plan/step15_mlp_full_results.md`
- Key result:
  - Swapping the encoder to `MLPDuelingDQN` hurt the baseline embedding on both primary `f1` metrics.
  - Both specialist branches improved versus the matched `MLP_baseline`.
  - `MLP_R5_volprof_for_return` was the strongest MLP branch:
    - it beat `MLP_baseline` on both `f1` metrics
    - it improved `f1` return over the official baseline
    - but it still missed the official baseline on `f1` Sharpe
- Next:
  - Keep the official `ConvDuelingDQN` baseline as the best balanced `f1` reference.
  - If this line continues, test a more targeted encoder change rather than another broad embedding sweep.

## 2026-04-13
- Summary: Backfilled the missing full `5`-seed `f1` mean for the Step 11 `D3_return_vol5` reduction branch.
- Changes:
  - Reused the existing Step 11 full runs for `seed42` and `seed43`.
  - Ran the missing `seed44`, `seed45`, and `seed46` full `f1` evaluations for `configs/signature_step11/d3_return_vol5.yaml`.
  - Merged all five seeds into:
    - `runs/step11_f1_reduction_full_mean/d3_return_vol5/`
  - Recorded the supplemental result in:
    - `docs/signature/plan/step11_reduction_full_mean_results.md`
- Key result:
  - `D3_return_vol5` full `5`-seed `f1` mean:
    - `Sharpe 0.8728`
    - `Return 95.3811%`

## 2026-04-13
- Summary: Completed the Step 16 `logsig.degree` specialist sweep and found no reason to move away from `degree=3`.
- Changes:
  - Added the `configs/signature_step16/` family for:
    - `L1_deg1`
    - `L2_deg2`
    - `L3_deg3`
    - `L4_deg4`
  - Ran the executable short `f1` degree sweep for:
    - `L2_deg2`
    - `L3_deg3`
    - `L4_deg4`
  - Recorded the result in:
    - `docs/signature/plan/step16_degree_short_results.md`
    - `docs/signature/plan/step16_degree_single_seed_full_results.md`
- Key result:
  - `degree=1` was structurally infeasible under the frozen `ConvDuelingDQN` backbone.
  - `degree=3` remained the strongest balanced short-run setting.
  - `degree=2` and `degree=4` showed only one-metric spikes and produced no promoted same-seed two-metric winner.
- Next:
  - Keep `degree=3` as the strongest known truncation level in the frozen baseline family.
  - If the degree line is revisited, open a larger-capacity branch instead of another local resweep.
  - versus the official baseline:
    - `Sharpe -0.0176`
    - `Return +9.5194 pct`
- Next:
  - Treat `D3_return_vol5` as a stronger `f1` specialist branch than previously established.
  - Keep the official baseline as the balanced default because `D3_return_vol5` still misses on Sharpe.
