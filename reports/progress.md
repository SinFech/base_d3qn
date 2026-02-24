# Progress Log

A running log of project progress that is easy for humans and agents to parse.

## Entries

### YYYY-MM-DD
- Summary:
- Changes:
- Blockers:
- Next:

### 2026-01-14
- Summary: Re-ran eval.py on baseline and signature variants for in-sample and OOS ranges using latest checkpoints.
- Changes: Updated eval outputs for baseline, signature-2, signature-3, and signature-4 runs; added OOS outputs under eval_oos_2021_2024; clarified that these results use sell_mode=all with no max_positions cap.
- Blockers: None.
- Next: Compare OOS vs in-sample metrics and decide follow-up experiments.

### 2026-01-17
- Summary: Trained sell-one variants, added eval position/action stats, and ran downtrend evaluations.
- Changes: New runs baseline_sell_one_e8a931, signature2_sell_one_83f00e, signature3_sell_one_9ba2a8 with sell_mode=one and max_positions=5; eval summaries now include position_stats; added eval.py trading_period override; ran 2018 (in-sample) and 2022 (OOS) downtrend evals with trading_period=200; trained rolling_vol and mean+vol signature runs with four evals each; trained sell-all signature runs for vol/mean/mean+vol embeddings with four evals each; trained sell_all_cap runs (max_positions=5) for baseline and signature 2/3 across 4 embeddings with four evals each.
- Blockers: None.
- Next: Review downtrend results and decide whether to retrain on bearish windows.

### 2026-01-18
- Summary: Added sell_one_plus mode and evaluated baseline + signature depth 2/3 across four embeddings.
- Changes: Added sell_mode=one_plus to the trading environment; eval action naming now distinguishes sell_one vs sell_all for 4-action agents; trained and evaluated one_plus runs (max_positions=5) across in-sample, OOS, and bear windows; logged results and action rates in experiments.md.
- Blockers: None.
- Next: Investigate why sell_one usage is near zero and consider reward/action-balance adjustments.

### 2026-01-20
- Summary: Ran OOS evals for existing sell-one checkpoints to compare with sell-one+.
- Changes: Added OOS 2021-2024 eval summaries for baseline_sell_one and signature sell_one runs; logged metrics in experiments.md.
- Blockers: None.
- Next: Compare sell-one vs sell-one+ in bear windows and decide if reward shaping is needed to increase sell usage.

### 2026-02-13
- Summary: Stabilized sr_enhanced training/evaluation metrics and refreshed signature experiment outputs.
- Changes: Added `sr_enhanced` support to train/eval CLI choices; fixed epsilon decay to global-step linear schedule; added periodic in-training eval (`eval_history.csv`) with fixed-window sampling and richer metrics; renamed episode metric field to `reward_return`; unified return-rate formula to `(equity_end / equity_start) - 1`; added eval diagnostics (`initial_state_none_episodes`, `zero_step_episodes`); fixed reward-overwrite bug in env so `profit/sr` rewards are no longer overwritten by hold shaping; aligned `trainer.evaluate()` episode length with configured `trading_period`; added percentage fields (`mean_return_rate_pct`, `std_return_rate_pct`, `median_return_rate_pct`) to `eval_summary.json`; added `configs/test_signature.yaml`; ran `testsignatueenhanced_e8a185` and regenerated eval artifacts.
- Blockers: Return-rate can still be very large under the current no-cash-constraint environment design (implicit leverage).
- Next: Introduce explicit cash/margin accounting in environment equity to constrain leverage and make return-rate magnitudes financially realistic.

### 2026-02-21
- Summary: Added PPO support on the same trading environment pipeline used by D3QN.
- Changes: Added `rl/algos/ppo/` with MLP actor-critic and PPO trainer (GAE, clipped objective, minibatch updates, checkpoints, metrics logging); added `scripts/train_ppo.py` and `scripts/eval_ppo.py`; added `configs/ppo_signature.yaml`; added compatibility so PPO loader can reuse legacy config fields from existing D3QN configs.
- Blockers: None.
- Next: Run full-length PPO in-sample and OOS comparisons against current D3QN baselines.

### 2026-02-22
- Summary: Upgraded PPO stack to continuous actions with capital-constrained accounting for industrial-style backtesting.
- Changes: Added `ContinuousTradingEnvironment` with explicit `cash/position/equity` ledger, transaction costs, slippage, and bankruptcy stop; extended `make_env` with `action_mode=continuous`; enabled signature observations to append account features; migrated PPO trainer to squashed-Gaussian continuous policy and added portfolio diagnostics in `metrics.csv`; updated PPO train/eval CLI with capital/cost overrides; refreshed `configs/ppo_signature.yaml` for continuous setup.
- Blockers: None.
- Next: Run multi-seed continuous PPO comparisons vs D3QN under the same train/val/test protocol and fixed OOS windows.

### 2026-02-22
- Summary: Added continuous SAC baseline on the same capital-constrained environment and completed first CPU benchmark run.
- Changes: Added `rl/algos/sac/` (Gaussian policy, twin Q critics, replay buffer, entropy-temperature tuning, soft target updates); added `scripts/train_sac.py` and `scripts/eval_sac.py`; added `configs/sac_signature.yaml`; ran smoke test (`smoke_sac_cont_cccac0`) and full 200-episode run (`sac_continuous_cpu_c4ef82`) with in-sample and OOS eval outputs.
- Blockers: Single-run OOS return-rate remains unstable and negative despite improved OOS reward metrics.
- Next: Run multi-seed SAC/PPO comparisons and tune SAC temperature/replay warmup for OOS robustness.

### 2026-02-23
- Summary: Added fractional discrete actions for capital-constrained D3QN and completed a 0.8-exposure action-space sweep.
- Changes: Extended `DiscreteCapitalTradingEnvironment` to support fractional buys (`buy_fractions`) and fractional sells (`sell_fractions`) with explicit cash/equity accounting; kept backward compatibility for legacy 3-action/4-action behavior; wired new env fields through `make_env`, D3QN trainer config/build paths, and `scripts/eval.py` action labeling/output.
- Blockers: Single-run OOS remains weak when action space is expanded aggressively (6/10/17 actions) under current reward setup.
- Next: Use multi-seed comparison for `0.8_3act` vs small fractional action sets, and retune reward/penalty shaping before further action-space expansion.

### 2026-02-23
- Summary: Ran and compared cash-constrained D3QN experiments for exposure cap and action granularity.
- Changes: Completed `max_exposure_ratio` sweep (`0.6`, `0.8`), then ran `0.8` with (1) multi-buy + sell-all (6 actions), (2) finer multi-buy + sell-all (10 actions), and (3) symmetric buy/sell portions (17 actions).
- Blockers: OOS Sharpe and return-rate degrade when moving from 3 actions to large fractional action spaces in single-seed runs.
- Next: Keep `0.8_3act` as control, run 3-5 seeds for reduced fractional action sets, and evaluate whether sell-side penalties/weights should be rebalanced.
