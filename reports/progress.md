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
