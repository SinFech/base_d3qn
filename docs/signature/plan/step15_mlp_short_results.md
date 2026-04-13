# Step 15 MLP Short-Run Results

## Scope

This note records the executed `S15_short_f1_mlp` short-stage results from Step 15.

Tested configs:

- `MLP_baseline`
- `MLP_D3_return_vol5`
- `MLP_R5_volprof_for_return`

All three configs switch the frozen D3QN encoder from `ConvDuelingDQN` to `MLPDuelingDQN`.

## Protocol

- Runner:
  - `scripts/walk_forward_protocol.py`
- Fold protocol:
  - `configs/folds_signature_step9_f1.json`
- Fold:
  - `f1` only
- Seeds:
  - `42,43,44`
- Short budget overrides:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- Eval episodes:
  - `50`
- Promotion rule:
  - promote candidate-seed pairs that beat the matched `MLP_baseline` seed on both:
    - `f1 oos_sharpe`
    - `f1 oos_return_pct`

## Mean-Level Short Results

| Config | `f1` OOS Sharpe mean | `f1` OOS Return % mean |
|---|---:|---:|
| `MLP_baseline` | `0.8279` | `68.2118` |
| `MLP_D3_return_vol5` | `0.8937` | `92.2096` |
| `MLP_R5_volprof_for_return` | `0.7760` | `96.5115` |

Mean-level deltas versus `MLP_baseline`:

- `MLP_D3_return_vol5`:
  - `Sharpe +0.0658`
  - `Return +23.9978 pct`
- `MLP_R5_volprof_for_return`:
  - `Sharpe -0.0519`
  - `Return +28.2996 pct`

## Same-Seed Promotion Check

Matched `MLP_baseline` seed controls:

- seed `42`: `Sharpe 0.7571`, `Return 59.7728`
- seed `43`: `Sharpe 0.6403`, `Return 70.0292`
- seed `44`: `Sharpe 1.0864`, `Return 74.8336`

Promoted candidate-seed pairs:

- `MLP_D3_return_vol5 seed42`
  - candidate: `Sharpe 0.8421`, `Return 111.0049`
  - delta vs baseline seed `42`: `+0.0850`, `+51.2321 pct`
- `MLP_D3_return_vol5 seed43`
  - candidate: `Sharpe 0.8995`, `Return 95.6427`
  - delta vs baseline seed `43`: `+0.2592`, `+25.6135 pct`
- `MLP_R5_volprof_for_return seed43`
  - candidate: `Sharpe 0.6507`, `Return 107.5679`
  - delta vs baseline seed `43`: `+0.0103`, `+37.5387 pct`

Rejected candidate-seed pairs:

- `MLP_D3_return_vol5 seed44`
  - Sharpe and Return both lost versus baseline seed `44`
- `MLP_R5_volprof_for_return seed42`
  - Sharpe improved only marginally, Return regressed
- `MLP_R5_volprof_for_return seed44`
  - Return improved strongly, but Sharpe stayed below baseline seed `44`

## Interpretation

- Switching to `MLPDuelingDQN` immediately helped the `D3_return_vol5` specialist branch in short `f1` runs.
- The `R5_volprof_for_return` branch became less consistent:
  - mean Sharpe fell below `MLP_baseline`
  - but one same-seed pair still satisfied the two-metric promotion rule
- Step 15 therefore promoted:
  - `MLP_D3_return_vol5`
  - `MLP_R5_volprof_for_return`

These promotions justify a full-stage MLP follow-up before accepting or rejecting the encoder-bottleneck hypothesis.

## Source files

- `docs/signature/plan/step15.md`
- `runs/step15_f1_mlp_short/mlp_baseline/results.csv`
- `runs/step15_f1_mlp_short/mlp_baseline/summary_by_algo.csv`
- `runs/step15_f1_mlp_short/mlp_d3_return_vol5/results.csv`
- `runs/step15_f1_mlp_short/mlp_d3_return_vol5/summary_by_algo.csv`
- `runs/step15_f1_mlp_short/mlp_r5_volprof_for_return/results.csv`
- `runs/step15_f1_mlp_short/mlp_r5_volprof_for_return/summary_by_algo.csv`
