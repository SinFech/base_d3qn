# Step 4 - Run Short Training and Stability Sweeps

## Goal

Run the short-budget `f3` screening sweep on the Step 3 survivors and decide which candidates are allowed to enter the expensive full walk-forward comparison.

## Scope

This step is limited to the short-run gate defined in `step2.md`.

It should:

- run the matched short-run control and survivor candidates from Step 3
- use the hardest fold `f3` only
- aggregate results across the frozen short-run seed set
- compare every candidate against the matched short-run baseline
- decide which candidates pass the `S4_short` gate and may continue to Step 5

It should not:

- reopen the candidate set
- use the official 5-seed full evaluation as a substitute for the short-run control
- make the final Step 6 adoption decision
- run the full walk-forward benchmark itself

## Inputs

- `docs/signature/plan/step2.md`
- `docs/signature/plan/step3_runtime_results.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/signature_step1/c1_std.yaml`
- `configs/signature_step1/c4_hlrange.yaml`
- `configs/folds_signature_step1_screen_f3.json`
- `scripts/walk_forward_protocol.py`

## Deliverables

- one short-run comparison record for the Step 3 survivors
- matched short-run baseline summary under the same budget
- aggregate candidate summaries across seeds `42,43,44`
- one explicit pass/fail decision for the `S4_short` gate
- one narrowed candidate set for Step 5

## Notes

### Frozen short-run protocol

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
- Device override:
  - `cuda`
- Output root:
  - `runs/step2_short_f3/`

### Gate rule

A candidate passes Step 4 only if, relative to the matched short-run baseline:

- it beats baseline on at least one primary metric on `f3`
- and it does not regress catastrophically on the other primary metric

The exact threshold values remain the ones frozen in `step2.md`.

## Evidence

- Short-run comparison record: `docs/signature/plan/step4_short_results.md`
- Raw short-run summaries: `runs/step2_short_f3/`
