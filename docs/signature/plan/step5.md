# Step 5 - Run Full Training and Evaluation on Final Candidates

## Goal

Run the full multi-fold, multi-seed walk-forward evaluation on the final Step 4 survivor set and compare it against the official full-run baseline.

## Scope

This step is limited to the full evaluation stage frozen in `step2.md`.

It should:

- run the final surviving candidate set under the full walk-forward protocol
- compare those results against the official baseline full-run
- report:
  - `f1`
  - `f2`
  - `f3`
  - overall OOS aggregate
- apply the Step 5 keep rule from `baseline.md`
- produce the full result table that Step 6 will use for the adoption recommendation

It should not:

- reopen the candidate shortlist decided by Steps 3-4
- introduce new training budgets for only one candidate
- treat short-run results as a substitute for the full walk-forward comparison
- make the final Step 6 recommendation by itself

## Inputs

- `docs/signature/plan/baseline.md`
- `docs/signature/plan/baseline_metrics.md`
- `docs/signature/plan/step2.md`
- `docs/signature/plan/step3_runtime_results.md`
- `docs/signature/plan/step4_short_results.md`
- baseline full-run artifacts:
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo.csv`
  - `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/summary_by_algo_fold.csv`
- candidate config:
  - `configs/signature_step1/c4_hlrange.yaml`
- fold protocol:
  - `configs/folds_rolling_long_oos.json`
- runner:
  - `scripts/walk_forward_protocol.py`

## Deliverables

- one full walk-forward result record for the final candidate set
- one matched comparison table against the official baseline on:
  - `f1`
  - `f2`
  - `f3`
  - overall OOS
- one explicit Step 5 keep/drop conclusion for the final candidate
- one narrowed evidence base for Step 6

## Notes

### Final candidate set

Step 4 reduced the full-evaluation set to:

- `baseline`
- `C4_hlrange`

### Baseline reuse rule

For Step 5, the official baseline full-run is reused as the matched baseline control:

- run group: `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`
- protocol:
  - folds from `configs/folds_rolling_long_oos.json`
  - seeds `42,43,44,45,46`
  - `3 folds x 5 seeds = 15 runs`

This reuse is acceptable because:

- it is the same frozen baseline config family
- it uses the same full-evaluation fold protocol
- it already serves as the official baseline source in `baseline.md`

### Frozen candidate run protocol

The new full run for `C4_hlrange` should use:

- config: `configs/signature_step1/c4_hlrange.yaml`
- folds: `configs/folds_rolling_long_oos.json`
- seeds: `42,43,44,45,46`
- train split override: `1.0`
- eval episodes: `50`
- eval epsilon: `0.0`
- the default training budget from the config family unless changed for every full-run config equally

### Comparison rule

The Step 5 keep rule remains the one frozen in `baseline.md`:

- the candidate is worth keeping if at least one target among:
  - `f1`
  - `f2`
  - `f3`
  - overall OOS
- beats the corresponding official baseline target on at least one primary metric:
  - higher `oos_sharpe_mean`, or
  - higher `oos_return_pct_mean`

### Ranking rule

If the candidate beats baseline on multiple targets, later Step 6 discussion should still rank the evidence in this order:

1. `worst_fold_oos_sharpe_mean`
2. overall `oos_sharpe_mean`
3. overall `oos_return_pct_mean`

## Evidence

- Step 4 survivor record: `docs/signature/plan/step4_short_results.md`
- Official baseline full-run: `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold/`
- Step 5 full-run result record: `docs/signature/plan/step5_full_results.md`
- Candidate full-run outputs: `runs/step5_full_c4_hlrange/`
