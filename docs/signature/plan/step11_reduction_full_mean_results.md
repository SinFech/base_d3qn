# Step 11 Reduction Full-Mean Supplement

## Scope

This note supplements Step 11 with the missing full `5`-seed `f1` mean for:

- `D3_return_vol5`

This config keeps:

- `log_return`
- `rolling_vol(window=5)`

and removes:

- `log_price`

## Protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - `42,43,44,45,46`
- Full budget family:
  - config default
  - effective default:
    - `train.num_episodes = 260`
    - `train.max_total_steps = 100000`
- Eval episodes:
  - `50`

Reuse policy:

- reused existing Step 11 full results for:
  - `seed42`
  - `seed43`
- newly ran the missing seeds:
  - `44`
  - `45`
  - `46`

Official baseline reference:

- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`

## Full `f1` Mean Comparison

| Config | `f1` OOS Sharpe mean | `f1` OOS Return % mean |
|---|---:|---:|
| official baseline | `0.8905` | `85.8617` |
| `D3_return_vol5` | `0.8728` | `95.3811` |
| delta (`candidate - baseline`) | `-0.0176` | `+9.5194` |

`D3_return_vol5` full-seed dispersion:

- `f1 oos_sharpe_std = 0.0593`
- `f1 oos_return_pct_std = 14.5882`

## Interpretation

- `D3_return_vol5` does keep the attractive `f1` return behavior when evaluated on all five seeds.
- The full mean no longer supports a strict two-metric win over the official baseline.
  - Return remains better than baseline.
  - Sharpe remains slightly below baseline.
- This makes `D3_return_vol5` more credible than a pure same-seed cherry pick, but still not strong enough to replace the official baseline for balanced `f1` performance.

## Updated Step 11 Read

The strongest evidence for `D3_return_vol5` is now:

- same-seed full win versus baseline seed `42`
- same-seed Sharpe-only win versus baseline seed `43`
- full `5`-seed mean:
  - return-positive versus official baseline
  - Sharpe-slightly-negative versus official baseline

Repository interpretation:

- keep `D3_return_vol5` as a serious `f1` specialist branch
- do not treat it as a new default
- read it as:
  - stronger than a fragile one-off cherry pick
  - still short of a clean full-mean replacement

## Source files

- `docs/signature/plan/step11.md`
- `docs/signature/plan/step11_reduction_single_seed_full_results.md`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- `runs/step11_f1_reduction_single_seed_full/d3_return_vol5_seed42/results.csv`
- `runs/step11_f1_reduction_single_seed_full/d3_return_vol5_seed43/results.csv`
- `runs/step11_f1_reduction_full_mean/d3_return_vol5_seed44/results.csv`
- `runs/step11_f1_reduction_full_mean/d3_return_vol5_seed45/results.csv`
- `runs/step11_f1_reduction_full_mean/d3_return_vol5_seed46/results.csv`
- `runs/step11_f1_reduction_full_mean/d3_return_vol5/summary_by_algo_fold.csv`
- `runs/step11_f1_reduction_full_mean/d3_return_vol5/summary_by_algo.csv`
