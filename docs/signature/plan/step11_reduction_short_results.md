# Step 11 Reduction Short-Screen Results

## Scope

This note records the executed `S11_short_f1_reduce` screen from Step 11.

The objective was intentionally narrow:

- `f1` only
- cherry-pick promotion only when the same short-run seed beats the matched baseline seed on both:
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

## Structural Feasibility Note

The one-channel reduction family was not executable under the frozen baseline backbone:

- `D4_price_only`
- `D5_return_only`
- `D6_vol5_only`

Reason:

- all three reduce the signature observation to `obs_dim = 9`
- the unchanged `ConvDuelingDQN` backbone then fails during network construction with:
  - `RuntimeError: Trying to create tensor with negative dimension -192`

Because Step 11 explicitly forbids changing the non-embedding parts of the baseline recipe, these candidates were recorded as structurally infeasible rather than repaired with a model-side change.

## Executed Reduction Results

| Candidate | Status | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs shared baseline | Delta Return % vs shared baseline | Promoted seeds |
|---|---|---:|---:|---:|---:|---|
| `D1_price_return` | executed | `0.7258` | `71.4326` | `-0.1667` | `-57.4371` | none |
| `D2_price_vol5` | executed | `0.7711` | `93.5407` | `-0.1215` | `-35.3290` | `44` |
| `D3_return_vol5` | executed | `0.9324` | `127.2506` | `+0.0398` | `-1.6191` | `42,43` |
| `D4_price_only` | infeasible | n/a | n/a | n/a | n/a | n/a |
| `D5_return_only` | infeasible | n/a | n/a | n/a | n/a | n/a |
| `D6_vol5_only` | infeasible | n/a | n/a | n/a | n/a | n/a |

## Interpretation

- Removing `log_price` while keeping `log_return + rolling_vol(window=5)` was the only reduction that improved mean `f1` Sharpe.
- `D3_return_vol5` still fell slightly short of the shared baseline on mean `f1` return, so its value was primarily a cherry-pick result rather than a clean mean-level replacement signal.
- `D2_price_vol5` was weak on the mean metrics, but `seed44` beat the matched short-run baseline on both primary `f1` metrics and therefore qualified for single-seed follow-up.
- The one-channel family did not reach training because the frozen backbone cannot accept `obs_dim = 9` without a model-side change.

## Promotion Decision

Promoted into `S11_single_seed_full`:

- `D2_price_vol5 seed44`
- `D3_return_vol5 seed42`
- `D3_return_vol5 seed43`

Not promoted:

- `D1_price_return`
- `D4_price_only`
- `D5_return_only`
- `D6_vol5_only`

## Source files

- `docs/signature/plan/step11.md`
- `docs/signature/plan/step9_f1_short_results.md`
- `runs/step9_f1_short/existing/baseline/results.csv`
- `runs/step9_f1_short/existing/baseline/summary_by_algo.csv`
- `runs/step11_f1_reduction_short/d1_price_return/results.csv`
- `runs/step11_f1_reduction_short/d1_price_return/summary_by_algo.csv`
- `runs/step11_f1_reduction_short/d2_price_vol5/results.csv`
- `runs/step11_f1_reduction_short/d2_price_vol5/summary_by_algo.csv`
- `runs/step11_f1_reduction_short/d3_return_vol5/results.csv`
- `runs/step11_f1_reduction_short/d3_return_vol5/summary_by_algo.csv`
- `runs/step11_f1_reduction_short/d4_price_only/launcher.log`
- `runs/step11_f1_reduction_short/d5_return_only/launcher.log`
- `runs/step11_f1_reduction_short/d6_vol5_only/launcher.log`
