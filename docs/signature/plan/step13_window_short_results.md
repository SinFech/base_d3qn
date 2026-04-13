# Step 13 Return-Vol Window Short-Screen Results

## Scope

This note records the executed `S13_short_f1_window` sweep from Step 13.

The family under test was:

- `log_return`
- `rolling_vol(window=k)`

with no `log_price` channel.

## Protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - `42,43,44`
- Short budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- Eval episodes:
  - `50`

Local family control:

- `DW5_return_vol5`

Reuse rule:

- `DW5_return_vol5` is identical in embedding to Step 11 `D3_return_vol5`
- its short-run result was reused directly from:
  - `runs/step11_f1_reduction_short/d3_return_vol5/`

Promotion rule:

- candidate-seed pair must beat the matched `DW5_return_vol5` seed on both:
  - `f1 oos_sharpe`
  - `f1 oos_return_pct`

## Short-Run Means

| Candidate | `rolling_vol.window` | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs `DW5` | Delta Return % vs `DW5` |
|---|---:|---:|---:|---:|---:|
| `DW5_return_vol5` | `5` | `0.9324` | `127.2506` | `0.0000` | `0.0000` |
| `DW20_return_vol20` | `20` | `0.8733` | `75.1748` | `-0.0591` | `-52.0758` |
| `DW3_return_vol3` | `3` | `0.8183` | `99.2355` | `-0.1141` | `-28.0151` |
| `DW10_return_vol10` | `10` | `0.7878` | `82.6008` | `-0.1446` | `-44.6497` |

## Same-Seed Gate Check

Matched `DW5_return_vol5` seeds:

- seed `42`: `SR 0.8968`, `Return 135.5084%`
- seed `43`: `SR 0.9341`, `Return 148.5770%`
- seed `44`: `SR 0.9663`, `Return 97.6662%`

Candidate-seed results:

| Candidate | Seed | Candidate `f1` Sharpe | Control `f1` Sharpe | Delta Sharpe | Candidate `f1` Return % | Control `f1` Return % | Delta Return % | Promote? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `DW3_return_vol3` | `42` | `0.6182` | `0.8968` | `-0.2786` | `67.9293` | `135.5084` | `-67.5791` | no |
| `DW3_return_vol3` | `43` | `0.9081` | `0.9341` | `-0.0260` | `144.1058` | `148.5770` | `-4.4712` | no |
| `DW3_return_vol3` | `44` | `0.9286` | `0.9663` | `-0.0377` | `85.6714` | `97.6662` | `-11.9949` | no |
| `DW10_return_vol10` | `42` | `0.5114` | `0.8968` | `-0.3854` | `77.4688` | `135.5084` | `-58.0396` | no |
| `DW10_return_vol10` | `43` | `0.8054` | `0.9341` | `-0.1287` | `95.2271` | `148.5770` | `-53.3499` | no |
| `DW10_return_vol10` | `44` | `1.0467` | `0.9663` | `+0.0804` | `75.1065` | `97.6662` | `-22.5597` | no |
| `DW20_return_vol20` | `42` | `0.9153` | `0.8968` | `+0.0185` | `132.3035` | `135.5084` | `-3.2049` | no |
| `DW20_return_vol20` | `43` | `0.8175` | `0.9341` | `-0.1166` | `38.0820` | `148.5770` | `-110.4950` | no |
| `DW20_return_vol20` | `44` | `0.8871` | `0.9663` | `-0.0792` | `55.1389` | `97.6662` | `-42.5273` | no |

## Interpretation

- No alternative window beat the `DW5_return_vol5` control on both same-seed primary metrics.
- `DW20_return_vol20 seed42` came closest:
  - Sharpe was slightly better
  - return was still lower by `3.2049` percentage points
- `DW10_return_vol10 seed44` produced the highest single-seed Sharpe in the sweep, but its return dropped too far to qualify.
- Within the `log_return + rolling_vol(window=k)` family, `window=5` remains the strongest executable horizon under this short-screen protocol.

## Promotion Decision

Promoted into `S13_single_seed_full`:

- none

Reason:

- no candidate-seed pair passed the Step 13 same-seed two-metric gate against `DW5_return_vol5`

## Source files

- `docs/signature/plan/step13.md`
- `docs/signature/plan/step11_reduction_short_results.md`
- `runs/step11_f1_reduction_short/d3_return_vol5/results.csv`
- `runs/step11_f1_reduction_short/d3_return_vol5/summary_by_algo.csv`
- `runs/step13_f1_return_vol_window_short/dw3_return_vol3/results.csv`
- `runs/step13_f1_return_vol_window_short/dw3_return_vol3/summary_by_algo.csv`
- `runs/step13_f1_return_vol_window_short/dw10_return_vol10/results.csv`
- `runs/step13_f1_return_vol_window_short/dw10_return_vol10/summary_by_algo.csv`
- `runs/step13_f1_return_vol_window_short/dw20_return_vol20/results.csv`
- `runs/step13_f1_return_vol_window_short/dw20_return_vol20/summary_by_algo.csv`
- `runs/step13_f1_return_vol_window_short/dw5_return_vol5/results.csv`
- `runs/step13_f1_return_vol_window_short/dw5_return_vol5/summary_by_algo.csv`
