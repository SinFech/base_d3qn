# Step 9 `f1`-Only Short-Screen Results

## Scope

This note records the executed `S9_short_f1` screen from Step 9.

The goal was to rank signature candidates only on the `f1` OOS targets:

- `f1 oos_sharpe_mean`
- `f1 oos_return_pct_mean`

No penalty was applied for `f2`, `f3`, or aggregate OOS behavior in this step.

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

Because the reopened candidate universe was large, the short screen was split into two matched batches:

- `existing`
  - baseline plus reopened implemented candidates
- `combos`
  - baseline plus the new Step 9 combination shortlist

Each batch kept its own matched baseline control.

## Batch A: `existing`

Matched baseline:

- `baseline`
  - `f1 oos_sharpe_mean = 0.8925`
  - `f1 oos_return_pct_mean = 128.8697`

Results:

| Candidate | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs batch baseline | Delta Return % vs batch baseline |
|---|---:|---:|---:|---:|
| `baseline` | `0.8925` | `128.8697` | `0.0000` | `0.0000` |
| `RV10` | `0.9131` | `84.0532` | `+0.0206` | `-44.8165` |
| `C6_deg4` | `0.9040` | `79.7812` | `+0.0114` | `-49.0885` |
| `RV20` | `0.9036` | `79.5530` | `+0.0110` | `-49.3166` |
| `C3_volprof` | `0.8710` | `66.6003` | `-0.0216` | `-62.2694` |
| `C5_multi` | `0.8486` | `97.4013` | `-0.0440` | `-31.4683` |
| `C2_bp` | `0.8442` | `96.9628` | `-0.0484` | `-31.9068` |
| `C4_hlrange` | `0.8342` | `106.8047` | `-0.0583` | `-22.0650` |
| `C1_std` | `0.8217` | `56.6999` | `-0.0708` | `-72.1698` |
| `B1_explore` | `0.6800` | `64.8236` | `-0.2125` | `-64.0460` |

Interpretation:

- No reopened singleton candidate passed the Step 9 short-screen pass rule.
- `RV10`, `C6_deg4`, and `RV20` improved Sharpe, but each gave up far too much `f1` return to qualify.
- `C4_hlrange` preserved more return than the Sharpe-oriented variants, but still did not beat the matched control on either metric.

## Batch B: `combos`

Matched baseline:

- `baseline`
  - `f1 oos_sharpe_mean = 0.8885`
  - `f1 oos_return_pct_mean = 72.2245`

Results:

| Candidate | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs batch baseline | Delta Return % vs batch baseline |
|---|---:|---:|---:|---:|
| `F1_hlrange_rv10` | `0.8957` | `91.5247` | `+0.0073` | `+19.3002` |
| `baseline` | `0.8885` | `72.2245` | `0.0000` | `0.0000` |
| `F1_std_rv10` | `0.8210` | `69.7954` | `-0.0675` | `-2.4291` |
| `F1_std_hlrange_rv10` | `0.8160` | `66.5855` | `-0.0725` | `-5.6390` |
| `F1_std_hlrange` | `0.7997` | `48.1886` | `-0.0887` | `-24.0359` |

Interpretation:

- `F1_hlrange_rv10` was the only Step 9 combination candidate that improved both `f1` primary metrics against its matched batch baseline.
- All standardization-based combinations underperformed their own control.

## Promotion Decision

Short-screen promotion outcome:

- Do not promote any `existing` singleton candidate.
- Promote only `F1_hlrange_rv10` into `S9_full_f1`.

Reason:

- It was the only candidate that clearly passed the Step 9 short-screen gate.
- It also ranked first within the combo batch under the Step 9 ordering:
  1. `f1 oos_sharpe_mean`
  2. `f1 oos_return_pct_mean`

## Notes

- The two short-screen baselines were intentionally treated as batch-local controls because Step 9 was split into separate batches.
- The short budget was noisy enough that a full `f1` follow-up remained necessary before writing the final Step 9 recommendation.

## Source files

- `runs/step9_f1_short/existing/baseline/summary_by_algo.csv`
- `runs/step9_f1_short/existing/c1_std/summary_by_algo.csv`
- `runs/step9_f1_short/existing/c2_bp/summary_by_algo.csv`
- `runs/step9_f1_short/existing/c3_volprof/summary_by_algo.csv`
- `runs/step9_f1_short/existing/c4_hlrange/summary_by_algo.csv`
- `runs/step9_f1_short/existing/c5_multi/summary_by_algo.csv`
- `runs/step9_f1_short/existing/c6_deg4/summary_by_algo.csv`
- `runs/step9_f1_short/existing/b1_explore/summary_by_algo.csv`
- `runs/step9_f1_short/existing/rv10/summary_by_algo.csv`
- `runs/step9_f1_short/existing/rv20/summary_by_algo.csv`
- `runs/step9_f1_short/combos/baseline/summary_by_algo.csv`
- `runs/step9_f1_short/combos/f1_std_hlrange/summary_by_algo.csv`
- `runs/step9_f1_short/combos/f1_std_rv10/summary_by_algo.csv`
- `runs/step9_f1_short/combos/f1_hlrange_rv10/summary_by_algo.csv`
- `runs/step9_f1_short/combos/f1_std_hlrange_rv10/summary_by_algo.csv`
