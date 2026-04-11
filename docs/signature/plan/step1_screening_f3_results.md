# Step 1 Screening Results on `f3`

_Recorded on 2026-04-11._

## Purpose

This note records a tractable first-pass experiment batch for the frozen Step 1 candidate set.

The goal of this batch is not to replace the official baseline in `baseline_metrics.md`.
It is a fast screening run to see which candidates are worth carrying forward after the Step 1 freeze.

## Screening protocol

- Candidate family:
  - `baseline`
  - `C1_std`
  - `C2_bp`
  - `C3_volprof`
  - `C4_hlrange`
  - `C5_multi`
  - `C6_deg4`
  - `B1_explore`
- Config roots:
  - baseline: `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
  - candidates: `configs/signature_step1/`
- Fold protocol: `configs/folds_signature_step1_screen_f3.json`
  - single fold: `f3`
  - train `2016-01-01 ~ 2020-12-31`
  - test `2021-01-01 ~ 2024-02-09`
- Seed: `42`
- Train overrides:
  - `train.num_episodes = 60`
  - `train.max_total_steps = 20000`
- Eval settings:
  - `eval.num_episodes = 50`
  - `eval.epsilon = 0.0`
  - fixed windows inherited from the configs
- Device override: `cuda`
- Output root: `runs/step1_screen_f3/`

## Important comparison caveat

This batch should be compared primarily against its own screening baseline control:

- `runs/step1_screen_f3/baseline`

It is also useful to view the official Step 0 `f3` baseline for context:

- official baseline source: `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- official `f3` reference:
  - `oos_sharpe_mean = -0.49653040461622266`
  - `oos_return_pct_mean = -17.583455920137908`

But the official baseline is a stronger protocol:

- `5 seeds`
- full walk-forward batch
- original training budget

So the official baseline should not be treated as a strict like-for-like ranking target for this screening pass.

## Screening baseline control

The matched screening control from `runs/step1_screen_f3/baseline` is:

- `oos_sharpe_mean = -0.2943528975881624`
- `oos_return_pct_mean = -9.16999563612066`
- `oos_reward_mean = -1.2618509422326034`

## Results

| Candidate | OOS Sharpe | OOS Return % | Delta Sharpe vs screening baseline | Delta Return % vs screening baseline | Interpretation |
|---|---:|---:|---:|---:|---|
| `baseline` | `-0.2944` | `-9.1700` | `0.0000` | `0.0000` | Matched screening control |
| `C1_std` | `-0.2719` | `-9.9566` | `+0.0225` | `-0.7866` | Slight Sharpe improvement only |
| `C2_bp` | `-0.3312` | `-11.7326` | `-0.0369` | `-2.5626` | Worse than screening baseline |
| `C3_volprof` | `-0.4789` | `-17.9700` | `-0.1846` | `-8.8000` | Clearly weaker on `f3` |
| `C4_hlrange` | `-0.2903` | `-13.0062` | `+0.0041` | `-3.8362` | Nearly flat Sharpe, worse return |
| `C5_multi` | `-0.4365` | `-16.3209` | `-0.1421` | `-7.1509` | Multi-scale alone did not help |
| `C6_deg4` | `-0.5123` | `-18.2711` | `-0.2179` | `-9.1011` | Depth 4 was negative in this pass |
| `B1_explore` | `-0.6343` | `-25.1362` | `-0.3400` | `-15.9662` | Bundle was the weakest candidate |

## Ranking by `f3` OOS Sharpe

1. `C1_std`
2. `C4_hlrange`
3. `baseline`
4. `C2_bp`
5. `C5_multi`
6. `C3_volprof`
7. `C6_deg4`
8. `B1_explore`

## Main findings

- No candidate beat the screening baseline on both primary metrics:
  - `oos_sharpe_mean`
  - `oos_return_pct_mean`
- `C1_std` was the only candidate with a noticeable positive Sharpe delta against the screening baseline, but it still lost on return.
- `C4_hlrange` was almost neutral on Sharpe, but still materially worse on return.
- `C6_deg4` underperformed both the screening baseline and the official Step 0 `f3` baseline.
- `B1_explore` was the weakest result in this pass, suggesting that stacking all exploratory feature changes at once is too aggressive for the current D3QN setup.

## Step 1 screening conclusion

If this screening pass is used as a first filter:

- keep `C1_std` as the most defensible carry-forward candidate
- optionally keep `C4_hlrange` as a weak secondary candidate if runtime budget allows
- drop `C2_bp`, `C3_volprof`, `C5_multi`, `C6_deg4`, and `B1_explore` from the next expensive benchmark stage unless there is a separate hypothesis for revisiting them

## Source files

- `runs/step1_screen_f3/baseline/summary_by_algo.csv`
- `runs/step1_screen_f3/c1_std/summary_by_algo.csv`
- `runs/step1_screen_f3/c2_bp/summary_by_algo.csv`
- `runs/step1_screen_f3/c3_volprof/summary_by_algo.csv`
- `runs/step1_screen_f3/c4_hlrange/summary_by_algo.csv`
- `runs/step1_screen_f3/c5_multi/summary_by_algo.csv`
- `runs/step1_screen_f3/c6_deg4/summary_by_algo.csv`
- `runs/step1_screen_f3/b1_explore/summary_by_algo.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
