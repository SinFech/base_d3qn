# Step 10 Single-Seed Full-Run Results

## Scope

This note records the executed Step 10 follow-up:

- promote short-run `(candidate, seed)` cherry picks
- run the corresponding single-seed full `f1` evaluation
- compare each promoted pair against the same-seed baseline full run

## Protocol

- runner:
  - `scripts/walk_forward_protocol.py`
- fold protocol:
  - `configs/folds_signature_step9_f1.json`
- fold:
  - `f1` only
- seeds:
  - one seed per run
- full budget family:
  - config default
  - effective default in this config family:
    - `train.num_episodes = 260`
    - `train.max_total_steps = 100000`
- eval episodes:
  - `50`

Promoted candidate-seed pairs:

| Candidate | Config | Seed | Reason promoted from Step 9 short run |
|---|---|---:|---|
| `C2_bp` | `configs/signature_step1/c2_bp.yaml` | `44` | same short-run seed beat the matched `existing` baseline on both `f1` metrics |
| `F1_std_rv10` | `configs/signature_step9/f1_std_rv10.yaml` | `42` | same short-run seed beat the matched `combos` baseline on both `f1` metrics |
| `F1_std_hlrange_rv10` | `configs/signature_step9/f1_std_hlrange_rv10.yaml` | `42` | same short-run seed beat the matched `combos` baseline on both `f1` metrics |
| `F1_hlrange_rv10` | `configs/signature_step9/f1_hlrange_rv10.yaml` | `43` | same short-run seed beat the matched `combos` baseline on both `f1` metrics |

Matched baseline controls:

- `baseline` seed `42`
- `baseline` seed `43`
- `baseline` seed `44`

## Results

### `C2_bp` seed `44` vs baseline seed `44`

| Config | `f1` OOS Sharpe | `f1` OOS Return % |
|---|---:|---:|
| baseline seed `44` | `0.8772` | `132.5071` |
| `C2_bp` seed `44` | `0.8799` | `94.4002` |
| delta (`candidate - baseline`) | `+0.0026` | `-38.1069` |

Interpretation:

- Sharpe gain survived only marginally.
- Return edge fully collapsed.

### `F1_std_rv10` seed `42` vs baseline seed `42`

| Config | `f1` OOS Sharpe | `f1` OOS Return % |
|---|---:|---:|
| baseline seed `42` | `0.7003` | `69.2750` |
| `F1_std_rv10` seed `42` | `0.8404` | `56.0134` |
| delta (`candidate - baseline`) | `+0.1400` | `-13.2616` |

Interpretation:

- Sharpe improvement survived.
- Return still lost by a wide margin.

### `F1_std_hlrange_rv10` seed `42` vs baseline seed `42`

| Config | `f1` OOS Sharpe | `f1` OOS Return % |
|---|---:|---:|
| baseline seed `42` | `0.7003` | `69.2750` |
| `F1_std_hlrange_rv10` seed `42` | `0.5508` | `37.3408` |
| delta (`candidate - baseline`) | `-0.1495` | `-31.9342` |

Interpretation:

- The short-run cherry-pick signal did not survive at all.
- Both primary metrics regressed materially.

### `F1_hlrange_rv10` seed `43` vs baseline seed `43`

| Config | `f1` OOS Sharpe | `f1` OOS Return % |
|---|---:|---:|
| baseline seed `43` | `0.8882` | `142.2286` |
| `F1_hlrange_rv10` seed `43` | `0.8498` | `124.9049` |
| delta (`candidate - baseline`) | `-0.0384` | `-17.3237` |

Interpretation:

- The strongest Step 9 cherry-pick candidate did not beat the same-seed baseline under full training.
- Both metrics regressed versus the matched control.

## Step 10 Conclusion

Step 10 result:

- no promoted short-run cherry-pick candidate survived as a same-seed two-metric full-run winner

What did survive:

- `C2_bp` seed `44` preserved a tiny Sharpe edge, but not return
- `F1_std_rv10` seed `42` preserved a larger Sharpe edge, but not return

What failed cleanly:

- `F1_std_hlrange_rv10` seed `42`
- `F1_hlrange_rv10` seed `43`

Repository interpretation:

- Step 9 short-run single-seed cherry picks were not robust enough to overturn the main multi-seed conclusion.
- No Step 10 candidate should be promoted beyond exploratory note status.

## Source files

- `docs/signature/plan/step10.md`
- `runs/step10_single_seed_f1/baseline_seed42/summary_by_algo.csv`
- `runs/step10_single_seed_f1/baseline_seed43/summary_by_algo.csv`
- `runs/step10_single_seed_f1/baseline_seed44/summary_by_algo.csv`
- `runs/step10_single_seed_f1/c2_bp_seed44/summary_by_algo.csv`
- `runs/step10_single_seed_f1/f1_std_rv10_seed42/summary_by_algo.csv`
- `runs/step10_single_seed_f1/f1_std_hlrange_rv10_seed42/summary_by_algo.csv`
- `runs/step10_single_seed_f1/f1_hlrange_rv10_seed43/summary_by_algo.csv`
