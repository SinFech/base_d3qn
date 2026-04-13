# Step 15 MLP Full Results

## Scope

This note records the executed `S15_full_f1_mlp` follow-up from Step 15.

Promoted configs from the short-stage MLP screen:

- `MLP_D3_return_vol5`
- `MLP_R5_volprof_for_return`

Shared MLP control:

- `MLP_baseline`

Official reference control:

- the frozen repository baseline `f1` `5`-seed average from:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`

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
  - effective default in this family:
    - `train.num_episodes = 260`
    - `train.max_total_steps = 100000`
- Eval episodes:
  - `50`
- Encoder change:
  - replace `ConvDuelingDQN` with `MLPDuelingDQN`

## Full `f1` Comparison

| Config | `f1` OOS Sharpe mean | `f1` OOS Return % mean |
|---|---:|---:|
| official baseline (`ConvDuelingDQN`) | `0.8905` | `85.8617` |
| `MLP_baseline` | `0.7437` | `77.2559` |
| `MLP_D3_return_vol5` | `0.8073` | `77.9949` |
| `MLP_R5_volprof_for_return` | `0.8059` | `100.1231` |

Delta versus the matched `MLP_baseline` control:

- `MLP_D3_return_vol5`:
  - `Sharpe +0.0636`
  - `Return +0.7391 pct`
- `MLP_R5_volprof_for_return`:
  - `Sharpe +0.0622`
  - `Return +22.8672 pct`

Delta versus the official repository baseline:

- `MLP_baseline`:
  - `Sharpe -0.1468`
  - `Return -8.6058 pct`
- `MLP_D3_return_vol5`:
  - `Sharpe -0.0832`
  - `Return -7.8667 pct`
- `MLP_R5_volprof_for_return`:
  - `Sharpe -0.0846`
  - `Return +14.2614 pct`

Dispersion across the five full seeds:

- `MLP_baseline`:
  - `f1 oos_sharpe_std = 0.0571`
  - `f1 oos_return_pct_std = 17.9191`
- `MLP_D3_return_vol5`:
  - `f1 oos_sharpe_std = 0.2025`
  - `f1 oos_return_pct_std = 21.5582`
- `MLP_R5_volprof_for_return`:
  - `f1 oos_sharpe_std = 0.0930`
  - `f1 oos_return_pct_std = 12.0410`

## Interpretation

- Replacing the encoder with `MLPDuelingDQN` did **not** improve the baseline embedding.
  - `MLP_baseline` regressed clearly versus the official `ConvDuelingDQN` baseline on both primary `f1` metrics.
- The main specialist branches did improve relative to the encoder-matched MLP control.
  - `MLP_D3_return_vol5` beat `MLP_baseline` on both full-run means.
  - `MLP_R5_volprof_for_return` also beat `MLP_baseline` on both full-run means and did so with lower dispersion than `MLP_D3_return_vol5`.
- The strongest Step 15 MLP branch was:
  - `MLP_R5_volprof_for_return`
  - It preserved nearly the same Sharpe level as `MLP_D3_return_vol5`
  - It added much stronger mean return
  - It was materially more stable across seeds
- Even so, no MLP candidate became a clean two-metric winner against the official repository baseline.
  - `MLP_R5_volprof_for_return` improved `f1` return versus the official baseline
  - but still missed on `f1` Sharpe

## Step 15 Decision

Step 15 outcome:

- The encoder-bottleneck hypothesis is **partially supported**.
- There is real encoder interaction:
  - `D3_return_vol5`
  - `R5_volprof_for_return`
  both became stronger relative to the MLP baseline than they were relative to the frozen Conv baseline.
- But a naive global swap from `ConvDuelingDQN` to `MLPDuelingDQN` is **not** the fix.
  - It hurt the baseline embedding materially.
  - It did not produce a new full-run two-metric winner over the official repository baseline.

Repository interpretation:

- Keep the official `ConvDuelingDQN` baseline as the best balanced `f1` reference.
- Keep `MLP_R5_volprof_for_return` as the strongest Step 15 evidence that the encoder interacts with signature design.
- If this diagnosis is pursued further, the next step should test:
  - split encoders for logsignature vs account features
  - or a more targeted encoder change
  rather than another blanket embedding sweep.

## Source files

- `docs/signature/plan/step15.md`
- `docs/signature/plan/step15_mlp_short_results.md`
- `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- `runs/step15_f1_mlp_full/mlp_baseline/results.csv`
- `runs/step15_f1_mlp_full/mlp_baseline/summary_by_algo.csv`
- `runs/step15_f1_mlp_full/mlp_d3_return_vol5/results.csv`
- `runs/step15_f1_mlp_full/mlp_d3_return_vol5/summary_by_algo.csv`
- `runs/step15_f1_mlp_full/mlp_r5_volprof_for_return/results.csv`
- `runs/step15_f1_mlp_full/mlp_r5_volprof_for_return/summary_by_algo.csv`
