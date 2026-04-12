# Step 2 - Define Benchmark Matrix and Acceptance Criteria

## Goal

Define the exact benchmark matrix, stage gates, and comparison rules that will be used to evaluate the Step 1 signature candidates against the frozen D3QN baseline.

## Scope

This step translates the Step 1 candidate freeze and the `f3` screening results into an executable benchmark plan for Steps 3-5.

It should:

- define which configs remain in the active benchmark matrix
- define which configs are dropped from the mainline after Step 1 screening
- freeze the benchmark stages used by later steps
- freeze the metrics, source files, and ranking order used for comparison
- define stage-specific acceptance criteria for runtime checks, short-run sweeps, and full evaluation

It should not:

- run the runtime benchmarks or training jobs itself
- introduce new candidate ideas beyond the Step 1 freeze
- revise the official baseline definition from `baseline.md`
- make the final adoption decision for the default signature recipe

## Inputs

- `docs/signature/plan/baseline.md`
- `docs/signature/plan/baseline_metrics.md`
- `docs/signature/plan/step1.md`
- `docs/signature/plan/step1_candidates.md`
- `docs/signature/plan/step1_screening_f3_results.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/signature_step1/`
- `configs/folds_rolling_long_oos.json`
- `configs/folds_signature_step1_screen_f3.json`
- `scripts/walk_forward_protocol.py`

## Deliverables

- one frozen benchmark matrix for Steps 3-5
- one active-candidate shortlist that later steps should use by default
- one dropped-candidate list that is out of the mainline unless scope is explicitly reopened
- one frozen set of stage-specific acceptance criteria
- one frozen ranking order for later result tables and Step 6 discussion

## Notes

### Baseline control rule

Every benchmark stage must include the matched baseline control:

- config: `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- candidate ID used in matrix tables: `baseline`

No candidate result should be interpreted without a baseline result from the same stage and budget.

### Active benchmark shortlist

Step 1 screening narrows the mainline benchmark set to:

| Tier | Candidate ID | Config | Reason |
|---|---|---|---|
| Control | `baseline` | `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml` | Required matched control in every stage |
| Primary | `C1_std` | `configs/signature_step1/c1_std.yaml` | Best `f3` Sharpe in Step 1 screening, with no observation-dimension increase |
| Secondary | `C4_hlrange` | `configs/signature_step1/c4_hlrange.yaml` | Near-baseline `f3` Sharpe and moderate feature expansion only |

### Candidates dropped from the mainline matrix

The following Step 1 candidates are not part of the default benchmark matrix after the `f3` screening pass:

- `C2_bp`
- `C3_volprof`
- `C5_multi`
- `C6_deg4`
- `B1_explore`

They may be revisited only if a later note explicitly reopens scope and explains why the Step 1 screening evidence is being overridden.

### Frozen benchmark stages

| Stage | Plan step | Purpose | Configs | Protocol | Main outputs |
|---|---|---|---|---|---|
| `S3_runtime` | Step 3 | Measure observation cost before more training | `baseline`, `C1_std`, `C4_hlrange` | repeated env/signature extraction under the same data windowing setup | runtime note plus recorded `obs_dim` and timing table |
| `S4_short` | Step 4 | Short-run robustness filter on the hardest fold | `baseline` plus Step 3 survivors | `f3` only, multiple seeds, short budget | matched summary tables against short-run baseline |
| `S5_full` | Step 5 | Full walk-forward comparison | `baseline` plus Step 4 survivors | `f1/f2/f3`, multi-seed, shared full budget | `summary_by_algo_fold.csv` and `summary_by_algo.csv` |

### Frozen stage protocols

#### `S3_runtime`

- Candidate set:
  - `baseline`
  - `C1_std`
  - `C4_hlrange`
- Required recorded fields:
  - `obs_dim`
  - env construction success/failure
  - signature observation extraction wall-clock time
  - relative timing ratio vs `baseline`
- Runtime benchmark should use the same observation path through `SignatureObsWrapper` as training uses.

#### `S4_short`

- Candidate set:
  - `baseline`
  - every candidate that passes `S3_runtime`
- Target fold:
  - `f3` only
- Seeds:
  - `42,43,44`
- Default short budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- Required comparison outputs:
  - per-seed results
  - mean and standard deviation across the short-run seeds
  - matched baseline short-run control under the same budget

#### `S5_full`

- Candidate set:
  - `baseline`
  - every candidate that passes `S4_short`
- Fold protocol:
  - `configs/folds_rolling_long_oos.json`
- Seeds:
  - `42,43,44,45,46`
- Default full budget:
  - keep the same training budget family as the frozen baseline unless a later step changes it for every surviving config equally
- Required outputs:
  - `summary_by_algo_fold.csv`
  - `summary_by_algo.csv`
  - explicit reporting for:
    - `f1`
    - `f2`
    - `f3`
    - overall OOS aggregate

### Frozen comparison metrics

Primary metrics:

- `oos_sharpe_mean`
- `oos_return_pct_mean`

Secondary diagnostic metrics:

- `oos_reward_mean`
- `oos_sharpe_std`
- `oos_return_pct_std`
- `is_sharpe_mean`
- `is_sharpe_std`

Stage-specific emphasis:

- `S3_runtime`: runtime ratio and `obs_dim`
- `S4_short`: `f3 oos_sharpe_mean` first, then `f3 oos_return_pct_mean`
- `S5_full`: `worst_fold_oos_sharpe_mean` first, then overall `oos_sharpe_mean`, then overall `oos_return_pct_mean`

### Frozen acceptance criteria

#### `S3_runtime` pass rule

A candidate may continue to `S4_short` only if:

- env/signature construction succeeds without shape or wrapper errors
- `obs_dim_ratio <= 2.0x` relative to `baseline`
- measured signature observation extraction time is not worse than `2.0x` the matched baseline control

#### `S4_short` pass rule

A candidate may continue to `S5_full` only if, against the matched short-run baseline:

- it beats baseline on at least one primary metric on `f3`:
  - higher `oos_sharpe_mean`, or
  - higher `oos_return_pct_mean`
- and it does not regress catastrophically on the other primary metric:
  - no worse than `0.05` Sharpe points, or
  - no worse than `5.0` return-percentage points

All seeds must also complete successfully with finite summary metrics.

#### `S5_full` keep rule

The full-evaluation keep rule remains the permissive rule from `baseline.md`:

- a candidate is worth keeping if at least one target among:
  - `f1`
  - `f2`
  - `f3`
  - overall OOS
- beats the corresponding official baseline target on at least one primary metric:
  - higher `oos_sharpe_mean`, or
  - higher `oos_return_pct_mean`

### Frozen ranking order for later discussion

When multiple surviving candidates are compared in later steps, rank them in this order:

1. `worst_fold_oos_sharpe_mean`
2. overall `oos_sharpe_mean`
3. overall `oos_return_pct_mean`

This ranking order reflects the current repository context:

- mean OOS performance matters
- but the main unresolved weakness for D3QN is still downside robustness on the hardest fold

## Evidence

- Official baseline and comparison rule: `docs/signature/plan/baseline.md`
- Step 1 candidate freeze: `docs/signature/plan/step1_candidates.md`
- Step 1 `f3` screening evidence: `docs/signature/plan/step1_screening_f3_results.md`
- Active candidate configs: `configs/signature_step1/`
