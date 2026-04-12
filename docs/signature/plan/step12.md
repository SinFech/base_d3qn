# Step 12 - Search `f1` Cherry Picks via Embedding Replacement

## Goal

Test whether a *replacement-style* embedding can produce stronger `f1` cherry-pick behavior than the current additive exploration path.

This step focuses on replacing baseline channels, not adding more channels on top of them.

## Scope

This step should:

- hold the baseline embedding size at three path channels
- replace exactly one baseline channel with one already-implemented alternative channel
- keep the evaluation target identical to Steps 9 and 10:
  - `f1` only
  - cherry-pick promotion only when the same short-run seed wins both `f1` metrics

This step should not:

- repeat additive candidates from Steps 1 and 9
- reopen rolling-vol window sweeps already covered by Steps 7 and 8
- change the non-embedding parts of the baseline recipe

## Inputs

- `docs/signature/plan/step9.md`
- `docs/signature/plan/step10.md`
- `docs/signature/plan/step10_single_seed_full_results.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/folds_signature_step9_f1.json`
- `scripts/walk_forward_protocol.py`

## Baseline Embedding for This Step

Current baseline path embedding:

- `log_price`
- `log_return`
- `rolling_vol(window=5)`

Allowed replacement channels in this step:

- `high_low_range`
- `normalized_cumulative_volume`

Not included here:

- `rolling_vol(window=10)`
- `rolling_vol(window=20)`

Reason:

- those horizon substitutions were already studied in Steps 7 and 8

## Frozen Replacement Candidate Set

Step 12 uses one-for-one replacements only.

### Replace one baseline channel with `high_low_range`

| Candidate ID | Path embedding | Replacement |
|---|---|---|
| `R1_hl_for_price` | `log_return + rolling_vol(window=5) + high_low_range` | replace `log_price` |
| `R2_hl_for_return` | `log_price + rolling_vol(window=5) + high_low_range` | replace `log_return` |
| `R3_hl_for_vol5` | `log_price + log_return + high_low_range` | replace `rolling_vol(window=5)` |

### Replace one baseline channel with `normalized_cumulative_volume`

| Candidate ID | Path embedding | Replacement |
|---|---|---|
| `R4_volprof_for_price` | `log_return + rolling_vol(window=5) + normalized_cumulative_volume` | replace `log_price` |
| `R5_volprof_for_return` | `log_price + rolling_vol(window=5) + normalized_cumulative_volume` | replace `log_return` |
| `R6_volprof_for_vol5` | `log_price + log_return + normalized_cumulative_volume` | replace `rolling_vol(window=5)` |

Control required for this step:

- baseline embedding with all three original channels

## Config Rule

Every Step 12 config must:

- start from the frozen baseline config family
- keep the same account features
- keep the same signature transform settings unless a later follow-up note explicitly changes them
- keep exactly three path channels
- differ from baseline only by a one-for-one embedding replacement

No additive fourth channel is allowed in this step.

## Protocol

### `S12_short_f1_replace`

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
- all Step 12 replacement candidates

### `S12_single_seed_full`

Promote only candidate-seed pairs where the same short-run seed beats the matched baseline on both:

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

Within promoted candidate-seed pairs, rank in this order:

1. same-seed delta `f1 oos_sharpe`
2. same-seed delta `f1 oos_return_pct`

## Deliverables

- one replacement config family:
  - `configs/signature_step12/`
- one short-run result note:
  - `docs/signature/plan/step12_replacement_short_results.md`
- one optional same-seed full result note:
  - `docs/signature/plan/step12_replacement_single_seed_full_results.md`
- one short-run output root:
  - `runs/step12_f1_replacement_short/`
- one optional full-run output root:
  - `runs/step12_f1_replacement_single_seed_full/`

## Outcome Type

Even if Step 12 finds a strong single-seed full-run winner, the result means:

- replacement-style embeddings deserve continued `f1` specialist attention

not:

- the repository-wide default should change
