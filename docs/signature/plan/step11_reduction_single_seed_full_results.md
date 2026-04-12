# Step 11 Reduction Single-Seed Full-Run Results

## Scope

This note records the executed `S11_single_seed_full` follow-up from Step 11.

Promoted candidate-seed pairs from the Step 11 short screen:

- `D2_price_vol5 seed44`
- `D3_return_vol5 seed42`
- `D3_return_vol5 seed43`

Shared matched baseline controls were reused from Step 10 because the protocol was identical:

- `baseline seed42`
- `baseline seed43`
- `baseline seed44`

## Protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - one seed per run
- Full budget family:
  - config default
  - effective default in this family:
    - `train.num_episodes = 260`
    - `train.max_total_steps = 100000`
- Eval episodes:
  - `50`

## Results

### `D2_price_vol5 seed44` vs baseline seed `44`

`D2_price_vol5` keeps:

- `log_price`
- `rolling_vol(window=5)`

and removes:

- `log_return`

| Config | `f1` OOS Sharpe | `f1` OOS Return % |
|---|---:|---:|
| baseline seed `44` | `0.8772` | `132.5071` |
| `D2_price_vol5` seed `44` | `0.7556` | `77.1807` |
| delta (`candidate - baseline`) | `-0.1217` | `-55.3263` |

Interpretation:

- The Step 11 short-run `seed44` cherry-pick signal did not survive.
- Both primary metrics regressed materially under full training.

### `D3_return_vol5 seed42` vs baseline seed `42`

`D3_return_vol5` keeps:

- `log_return`
- `rolling_vol(window=5)`

and removes:

- `log_price`

| Config | `f1` OOS Sharpe | `f1` OOS Return % |
|---|---:|---:|
| baseline seed `42` | `0.7003` | `69.2750` |
| `D3_return_vol5` seed `42` | `0.8227` | `108.3174` |
| delta (`candidate - baseline`) | `+0.1224` | `+39.0424` |

Interpretation:

- This cherry-pick signal survived cleanly under full training.
- `D3_return_vol5 seed42` remained a strict same-seed two-metric winner.

### `D3_return_vol5 seed43` vs baseline seed `43`

| Config | `f1` OOS Sharpe | `f1` OOS Return % |
|---|---:|---:|
| baseline seed `43` | `0.8882` | `142.2286` |
| `D3_return_vol5` seed `43` | `0.9157` | `103.6459` |
| delta (`candidate - baseline`) | `+0.0275` | `-38.5827` |

Interpretation:

- The Sharpe edge survived.
- The return edge did not.
- `D3_return_vol5 seed43` is therefore not a same-seed two-metric full-run winner.

## Step 11 Decision

Step 11 outcome:

- `D3_return_vol5 seed42` is the strongest reduction-style `f1` cherry-pick discovered in this step.
- `D3_return_vol5 seed43` preserved only Sharpe upside.
- `D2_price_vol5 seed44` failed cleanly under full follow-up.

Repository interpretation:

- Keep `D3_return_vol5 seed42` as a valid reduction-style `f1` specialist note.
- Do not treat Step 11 as a new repository-wide default decision.
- Treat all other Step 11 promoted pairs as failed or partial cherry-picks rather than stable upgrades.

## Source files

- `docs/signature/plan/step11.md`
- `docs/signature/plan/step11_reduction_short_results.md`
- `docs/signature/plan/step10_single_seed_full_results.md`
- `runs/step10_single_seed_f1/baseline_seed42/summary_by_algo.csv`
- `runs/step10_single_seed_f1/baseline_seed43/summary_by_algo.csv`
- `runs/step10_single_seed_f1/baseline_seed44/summary_by_algo.csv`
- `runs/step11_f1_reduction_single_seed_full/d2_price_vol5_seed44/summary_by_algo.csv`
- `runs/step11_f1_reduction_single_seed_full/d3_return_vol5_seed42/summary_by_algo.csv`
- `runs/step11_f1_reduction_single_seed_full/d3_return_vol5_seed43/summary_by_algo.csv`
