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
