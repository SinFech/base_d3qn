# Step 4 Short-Run Results on `f3`

_Recorded on 2026-04-11._

## Protocol

- Candidates:
  - `baseline`
  - `C1_std`
  - `C4_hlrange`
- Fold protocol:
  - `configs/folds_signature_step1_screen_f3.json`
- Fold:
  - `f3`
- Seeds:
  - `42,43,44`
- Train overrides:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- Eval settings:
  - `eval.num_episodes = 50`
  - `eval.epsilon = 0.0`
- Device:
  - `cuda`
- Output root:
  - `runs/step2_short_f3/`

## Matched short-run baseline

The matched short-run control is:

- `oos_sharpe_mean = -0.49012962810355715`
- `oos_return_pct_mean = -16.760331446246123`
- `oos_sharpe_std = 0.24822566302703172`
- `oos_return_pct_std = 2.9709846699941265`

This is close to the official Step 0 `f3` baseline:

- official `f3 oos_sharpe_mean = -0.49653040461622266`
- official `f3 oos_return_pct_mean = -17.583455920137908`

So the short-run control is a credible gate for this stage.

## Results

| Candidate | `oos_sharpe_mean` | `oos_return_pct_mean` | Delta Sharpe vs short baseline | Delta Return % vs short baseline | S4 gate |
|---|---:|---:|---:|---:|---|
| `baseline` | `-0.4901` | `-16.7603` | `0.0000` | `0.0000` | Control |
| `C1_std` | `-0.5906` | `-16.2425` | `-0.1005` | `+0.5178` | Fail |
| `C4_hlrange` | `-0.2975` | `-13.5065` | `+0.1927` | `+3.2538` | Pass |

## Interpretation

- `C1_std` improved return slightly, but its Sharpe degraded by more than the allowed `0.05` threshold.
- `C4_hlrange` improved both primary metrics against the matched short-run baseline and clearly passed the gate.
- The short-run evidence says `high_low_range` is the only Step 2 shortlist candidate worth carrying into the full evaluation stage.

## Step 4 conclusion

The Step 5 candidate set is:

- `baseline`
- `C4_hlrange`

`C1_std` is dropped from the mainline after the short-run gate.

## Source files

- `runs/step2_short_f3/baseline/summary_by_algo.csv`
- `runs/step2_short_f3/c1_std/summary_by_algo.csv`
- `runs/step2_short_f3/c4_hlrange/summary_by_algo.csv`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
