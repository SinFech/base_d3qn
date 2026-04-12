# Step 12 Replacement Short-Screen Results

## Scope

This note records the executed `S12_short_f1_replace` screen from Step 12.

The objective stayed aligned with Step 10:

- `f1` only
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
| `R1_hl_for_price` | replace `log_price` with `high_low_range` | `0.7980` | `95.6912` | `-0.0946` | `-33.1784` | none |
| `R2_hl_for_return` | replace `log_return` with `high_low_range` | `0.8577` | `90.0432` | `-0.0348` | `-38.8265` | none |
| `R3_hl_for_vol5` | replace `rolling_vol(window=5)` with `high_low_range` | `0.8791` | `115.3303` | `-0.0135` | `-13.5393` | none |
| `R4_volprof_for_price` | replace `log_price` with `normalized_cumulative_volume` | `0.8873` | `117.5257` | `-0.0052` | `-11.3440` | none |
| `R5_volprof_for_return` | replace `log_return` with `normalized_cumulative_volume` | `0.8380` | `90.6803` | `-0.0546` | `-38.1894` | `42` |
| `R6_volprof_for_vol5` | replace `rolling_vol(window=5)` with `normalized_cumulative_volume` | `0.8761` | `82.5308` | `-0.0165` | `-46.3389` | none |

## Interpretation

- No replacement candidate beat the shared baseline on both short-run mean metrics.
- The closest mean-level challenger was `R4_volprof_for_price`, which nearly matched the shared baseline on Sharpe while still losing on return.
- `R5_volprof_for_return` was the only replacement candidate that produced a valid same-seed two-metric cherry pick, and it did so only on `seed42`.
- Replacing `rolling_vol(window=5)` with either `high_low_range` or `normalized_cumulative_volume` did not produce a strong `f1` signal in this short-run protocol.

## Promotion Decision

Promoted into `S12_single_seed_full`:

- `R5_volprof_for_return seed42`

Not promoted:

- `R1_hl_for_price`
- `R2_hl_for_return`
- `R3_hl_for_vol5`
- `R4_volprof_for_price`
- `R6_volprof_for_vol5`

## Source files

- `docs/signature/plan/step12.md`
- `docs/signature/plan/step9_f1_short_results.md`
- `runs/step9_f1_short/existing/baseline/results.csv`
- `runs/step9_f1_short/existing/baseline/summary_by_algo.csv`
- `runs/step12_f1_replacement_short/r1_hl_for_price/results.csv`
- `runs/step12_f1_replacement_short/r1_hl_for_price/summary_by_algo.csv`
- `runs/step12_f1_replacement_short/r2_hl_for_return/results.csv`
- `runs/step12_f1_replacement_short/r2_hl_for_return/summary_by_algo.csv`
- `runs/step12_f1_replacement_short/r3_hl_for_vol5/results.csv`
- `runs/step12_f1_replacement_short/r3_hl_for_vol5/summary_by_algo.csv`
- `runs/step12_f1_replacement_short/r4_volprof_for_price/results.csv`
- `runs/step12_f1_replacement_short/r4_volprof_for_price/summary_by_algo.csv`
- `runs/step12_f1_replacement_short/r5_volprof_for_return/results.csv`
- `runs/step12_f1_replacement_short/r5_volprof_for_return/summary_by_algo.csv`
- `runs/step12_f1_replacement_short/r6_volprof_for_vol5/results.csv`
- `runs/step12_f1_replacement_short/r6_volprof_for_vol5/summary_by_algo.csv`
