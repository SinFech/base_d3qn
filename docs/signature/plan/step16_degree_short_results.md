# Step 16 Degree Short-Screen Results

## Scope

This note records the executed `S16_short_f1_degree` screen from Step 16.

The objective stayed aligned with the current `f1` specialist line:

- `f1` only
- fixed baseline embedding
- fixed `ConvDuelingDQN` encoder
- vary only `signature.logsig.degree`
- promote only same-seed short-run wins that beat the matched `L3_deg3` seed on both:
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

## Structural Feasibility Check

Step 16 defined four degree candidates:

- `L1_deg1`
- `L2_deg2`
- `L3_deg3`
- `L4_deg4`

But `L1_deg1` was structurally infeasible under the frozen `ConvDuelingDQN` backbone:

- `L1_deg1` observation size:
  - `obs_dim = 8`
- failure mode:
  - the current conv stack produces a negative hidden dimension when `obs_dim` is that small
- exact construction error:
  - `RuntimeError: Trying to create tensor with negative dimension -256: [120, -256]`

So the executed short batch contained only:

- `L2_deg2`
- `L3_deg3`
- `L4_deg4`

## Mean-Level Short Results

| Candidate | `logsig.degree` | `f1` OOS Sharpe mean | `f1` OOS Return % mean | Delta Sharpe vs `L3_deg3` | Delta Return % vs `L3_deg3` | Promoted seeds |
|---|---:|---:|---:|---:|---:|---|
| `L2_deg2` | `2` | `0.7435` | `94.9931` | `-0.1724` | `-5.5530` | none |
| `L3_deg3` | `3` | `0.9159` | `100.5461` | `0.0000` | `0.0000` | control |
| `L4_deg4` | `4` | `0.8841` | `96.9654` | `-0.0319` | `-3.5807` | none |

Mean-level ranking:

1. `L3_deg3`
2. `L4_deg4`
3. `L2_deg2`

## Same-Seed Gate Check

Matched `L3_deg3` seed controls:

- seed `42`:
  - `f1 oos_sharpe = 0.8825`
  - `f1 oos_return_pct = 72.2423`
- seed `43`:
  - `f1 oos_sharpe = 0.8292`
  - `f1 oos_return_pct = 118.0572`
- seed `44`:
  - `f1 oos_sharpe = 1.0360`
  - `f1 oos_return_pct = 111.3388`

Candidate comparisons:

| Candidate | Seed | Candidate `f1` Sharpe | `L3_deg3` `f1` Sharpe | Delta Sharpe | Candidate `f1` Return % | `L3_deg3` `f1` Return % | Delta Return % | Promote? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `L2_deg2` | `42` | `0.8471` | `0.8825` | `-0.0354` | `129.0202` | `72.2423` | `+56.7779` | no |
| `L2_deg2` | `43` | `0.7619` | `0.8292` | `-0.0673` | `79.4793` | `118.0572` | `-38.5779` | no |
| `L2_deg2` | `44` | `0.6214` | `1.0360` | `-0.4146` | `76.4798` | `111.3388` | `-34.8590` | no |
| `L4_deg4` | `42` | `0.7745` | `0.8825` | `-0.1080` | `55.6628` | `72.2423` | `-16.5794` | no |
| `L4_deg4` | `43` | `0.9017` | `0.8292` | `+0.0725` | `99.5091` | `118.0572` | `-18.5481` | no |
| `L4_deg4` | `44` | `0.9760` | `1.0360` | `-0.0600` | `135.7244` | `111.3388` | `+24.3856` | no |

## Interpretation

- `degree=3` remains the strongest balanced short-run setting inside the frozen baseline family.
- `degree=2` produced one strong return-only spike:
  - `L2_deg2 seed42`
  - but it lost on Sharpe
- `degree=4` produced two partial specialist hints:
  - `L4_deg4 seed43` improved Sharpe only
  - `L4_deg4 seed44` improved Return only
- No alternative degree produced a same-seed two-metric short-run win over `L3_deg3`.

## Promotion Decision

Promoted into `S16_single_seed_full`:

- none

Reason:

- no candidate-seed pair beat the matched `L3_deg3` seed on both Step 16 primary metrics

## Source files

- `docs/signature/plan/step16.md`
- `runs/step16_f1_degree_short/l2_deg2/results.csv`
- `runs/step16_f1_degree_short/l2_deg2/summary_by_algo.csv`
- `runs/step16_f1_degree_short/l3_deg3/results.csv`
- `runs/step16_f1_degree_short/l3_deg3/summary_by_algo.csv`
- `runs/step16_f1_degree_short/l4_deg4/results.csv`
- `runs/step16_f1_degree_short/l4_deg4/summary_by_algo.csv`
