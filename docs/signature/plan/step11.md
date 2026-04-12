# Step 11 - Search `f1` Cherry Picks via Embedding Reduction

## Goal

Test whether a *smaller* signature embedding than the current baseline can produce stronger `f1` single-seed cherry-pick behavior.

This step keeps the objective intentionally narrow:

- only `f1`
- only the two primary `f1` metrics
- only cherry-pick candidate-seed pairs that win both metrics in short runs

## Scope

This step should:

- test strict *reductions* of the baseline path embedding
- avoid adding any new path channel
- avoid replacing a baseline channel with a new channel type
- use the same Step 10 cherry-pick promotion logic:
  - short-run candidate-seed pair must beat the matched baseline seed on both `f1` metrics before promotion

This step should not:

- reinterpret multi-seed results from Steps 9 and 10
- change the default baseline recipe
- widen scope beyond signature-path embedding changes

## Inputs

- `docs/signature/plan/step9.md`
- `docs/signature/plan/step10.md`
- `docs/signature/plan/step10_single_seed_full_results.md`
- `docs/signature/plan/baseline_metrics.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/folds_signature_step9_f1.json`
- `scripts/walk_forward_protocol.py`

## Baseline Embedding for This Step

Current baseline path embedding:

- `log_price`
- `log_return`
- `rolling_vol(window=5)`

Step 11 asks only:

- what happens if one or two of those channels are removed?

## Frozen Reduction Candidate Set

This step uses the full strict-subset family of the baseline embedding.

| Candidate ID | Path embedding | Reduction type |
|---|---|---|
| `D1_price_return` | `log_price + log_return` | remove `rolling_vol(window=5)` |
| `D2_price_vol5` | `log_price + rolling_vol(window=5)` | remove `log_return` |
| `D3_return_vol5` | `log_return + rolling_vol(window=5)` | remove `log_price` |
| `D4_price_only` | `log_price` | keep one channel only |
| `D5_return_only` | `log_return` | keep one channel only |
| `D6_vol5_only` | `rolling_vol(window=5)` | keep one channel only |

Control required for this step:

- baseline embedding with all three channels

## Config Rule

Every Step 11 config must:

- start from the frozen baseline config family
- keep the same account features
- keep the same signature transform settings unless explicitly changed later by a follow-up note
- change only the `embedding` channel list

No new path channel is allowed in this step.

## Protocol

### `S11_short_f1_reduce`

Run one short `f1`-only screen:

- fold protocol:
  - `configs/folds_signature_step9_f1.json`
- fold:
  - `f1` only
- seeds:
  - `42,43,44`
- short budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- eval episodes:
  - `50`

Batch composition:

- baseline
- all Step 11 reduction candidates

### `S11_single_seed_full`

Promote only candidate-seed pairs that satisfy the same Step 10 cherry-pick rule:

- same short-run seed beats the matched batch baseline on both:
  - `f1 oos_sharpe`
  - `f1 oos_return_pct`

Then run:

- one single-seed full `f1` evaluation for the promoted candidate-seed pair
- one same-seed baseline full control

Full budget family:

- config default
- effective baseline family default:
  - `train.num_episodes = 260`
  - `train.max_total_steps = 100000`

## Ranking Rule

This step is diagnostic and cherry-pick-oriented.

Rank promoted results in this order:

1. same-seed delta `f1 oos_sharpe`
2. same-seed delta `f1 oos_return_pct`

## Deliverables

- one reduction config family:
  - `configs/signature_step11/`
- one short-run result note:
  - `docs/signature/plan/step11_reduction_short_results.md`
- one optional same-seed full result note:
  - `docs/signature/plan/step11_reduction_single_seed_full_results.md`
- one short-run output root:
  - `runs/step11_f1_reduction_short/`
- one optional full-run output root:
  - `runs/step11_f1_reduction_single_seed_full/`

## Outcome Type

Even if Step 11 finds a good seed-specific winner, the interpretation remains:

- an `f1` cherry-pick branch for reduced embeddings

not:

- a new repository-wide default
