# Step 7 Short Rolling-Vol Window Sweep Results

_Recorded on 2026-04-11._

## Protocol

- Candidate family:
  - `RV3`
  - `RV5`
  - `RV10`
  - `RV20`
- Config directory:
  - `configs/signature_step7/`
- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step1_screen_f3.json`
- Fold:
  - `f3` only
- Seeds:
  - `42,43,44`
- Train budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- Eval episodes:
  - `50`
- Output root:
  - `runs/step7_short_volwindow_f3/`

## Sweep table

| Candidate | `rolling_vol.window` | OOS Sharpe mean | OOS Return % mean | Delta Sharpe vs `RV5` | Delta Return % vs `RV5` | Decision |
|---|---:|---:|---:|---:|---:|---|
| `RV3` | `3` | `-0.4353` | `-15.7772` | `-0.0480` | `-1.0056` | Drop |
| `RV5` | `5` | `-0.3872` | `-14.7716` | `0.0000` | `0.0000` | Control |
| `RV10` | `10` | `-0.3086` | `-12.0602` | `+0.0786` | `+2.7114` | Promote to full |
| `RV20` | `20` | `-0.3214` | `-12.9860` | `+0.0659` | `+1.7856` | Keep below `RV10` |

## Ranking

Using the Step 7 short-run gate and the same `f3`-first logic as Step 4, the short-sweep ranking is:

1. `RV10`
2. `RV20`
3. `RV5`
4. `RV3`

## Interpretation

- `RV3` underperformed the matched `RV5` control on both primary metrics and is closed.
- `RV20` improved both primary metrics over `RV5`, so it passed the short gate, but it was weaker than `RV10`.
- `RV10` was the strongest non-default window on both primary metrics:
  - Sharpe improved from `-0.3872` to `-0.3086`
  - Return improved from `-14.7716%` to `-12.0602%`

## Step 7 short-stage decision

- Promote exactly one non-default window to the full walk-forward stage:
  - `RV10`
- Do not promote `RV20` because Step 7 allows at most one non-default survivor and `RV10` ranked higher on both primary metrics.

## Source files

- `runs/step7_short_volwindow_f3/rv3/summary_by_algo.csv`
- `runs/step7_short_volwindow_f3/rv5/summary_by_algo.csv`
- `runs/step7_short_volwindow_f3/rv10/summary_by_algo.csv`
- `runs/step7_short_volwindow_f3/rv20/summary_by_algo.csv`
