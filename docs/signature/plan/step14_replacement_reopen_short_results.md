# Step 14 Replacement Reopen Short-Screen Results

## Scope

This note records the executed `S14_short_f1_replace_reopen` screen from Step 14.

The objective stayed aligned with the existing `f1` specialist line:

- `f1` only
- replacement-only candidates
- promote only same-seed short-run cherry picks that beat the matched baseline seed on both:
  - `f1 oos_sharpe`
  - `f1 oos_return_pct`

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

Shared short-run baseline control reused from Step 9:

- source:
  - `runs/step9_f1_short/existing/baseline/`
- baseline mean:
  - `f1 oos_sharpe_mean = 0.8925`
  - `f1 oos_return_pct_mean = 128.8697`

Matched baseline seeds used for same-seed cherry-pick checks:

- seed `42`:
  - `f1 oos_sharpe = 0.8696`
  - `f1 oos_return_pct = 125.7058`
- seed `43`:
  - `f1 oos_sharpe = 0.9227`
  - `f1 oos_return_pct = 127.4294`
- seed `44`:
  - `f1 oos_sharpe = 0.8853`
  - `f1 oos_return_pct = 133.4738`

## Replacement Results

| Candidate | Replacement | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs shared baseline | Delta Return % vs shared baseline | Promoted seeds |
|---|---|---:|---:|---:|---:|---|
| `RM1_mean_for_price` | replace `log_price` with `rolling_mean(window=5)` | `0.7813` | `91.2854` | `-0.1112` | `-37.5843` | none |
| `RM2_mean_for_return` | replace `log_return` with `rolling_mean(window=5)` | `0.8776` | `92.5916` | `-0.0149` | `-36.2780` | none |
| `RM3_mean_for_vol5` | replace `rolling_vol(window=5)` with `rolling_mean(window=5)` | `0.7480` | `84.1296` | `-0.1445` | `-44.7401` | none |

## Same-Seed Gate Check

| Candidate | Seed | Candidate `f1` Sharpe | Baseline `f1` Sharpe | Delta Sharpe | Candidate `f1` Return % | Baseline `f1` Return % | Delta Return % | Promote? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `RM1_mean_for_price` | `42` | `0.6448` | `0.8696` | `-0.2248` | `45.5558` | `125.7058` | `-80.1500` | no |
| `RM1_mean_for_price` | `43` | `0.8259` | `0.9227` | `-0.0968` | `102.8134` | `127.4294` | `-24.6160` | no |
| `RM1_mean_for_price` | `44` | `0.8733` | `0.8853` | `-0.0120` | `125.4870` | `133.4738` | `-7.9868` | no |
| `RM2_mean_for_return` | `42` | `0.7103` | `0.8696` | `-0.1594` | `75.8682` | `125.7058` | `-49.8376` | no |
| `RM2_mean_for_return` | `43` | `1.0504` | `0.9227` | `+0.1277` | `75.4931` | `127.4294` | `-51.9363` | no |
| `RM2_mean_for_return` | `44` | `0.8722` | `0.8853` | `-0.0131` | `126.4136` | `133.4738` | `-7.0602` | no |
| `RM3_mean_for_vol5` | `42` | `0.7337` | `0.8696` | `-0.1360` | `106.3668` | `125.7058` | `-19.3390` | no |
| `RM3_mean_for_vol5` | `43` | `0.8111` | `0.9227` | `-0.1116` | `50.7891` | `127.4294` | `-76.6404` | no |
| `RM3_mean_for_vol5` | `44` | `0.6994` | `0.8853` | `-0.1859` | `95.2330` | `133.4738` | `-38.2408` | no |

## Interpretation

- No `rolling_mean` replacement candidate beat the shared Step 9 baseline on both mean metrics.
- `RM2_mean_for_return` was the closest mean-level challenger on Sharpe, but it still lost badly on return.
- No same-seed gate passed:
  - `RM2 seed43` had the strongest Sharpe-only spike
  - `RM2 seed44` came closest to a balanced same-seed comparison
  - neither candidate-seed pair beat the matched baseline on both primary metrics
- Reopening the missing `rolling_mean` family did not uncover a Step 12-style replacement winner.

## Promotion Decision

Promoted into `S14_single_seed_full`:

- none

Reason:

- no candidate-seed pair passed the Step 14 same-seed two-metric gate against the shared baseline control

## Source files

- `docs/signature/plan/step14.md`
- `docs/signature/plan/step9_f1_short_results.md`
- `runs/step9_f1_short/existing/baseline/results.csv`
- `runs/step9_f1_short/existing/baseline/summary_by_algo.csv`
- `runs/step14_f1_replacement_reopen_short/rm1_mean_for_price/results.csv`
- `runs/step14_f1_replacement_reopen_short/rm1_mean_for_price/summary_by_algo.csv`
- `runs/step14_f1_replacement_reopen_short/rm2_mean_for_return/results.csv`
- `runs/step14_f1_replacement_reopen_short/rm2_mean_for_return/summary_by_algo.csv`
- `runs/step14_f1_replacement_reopen_short/rm3_mean_for_vol5/results.csv`
- `runs/step14_f1_replacement_reopen_short/rm3_mean_for_vol5/summary_by_algo.csv`
