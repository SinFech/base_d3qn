# Step 1 - Freeze Candidate Signature Upgrade Set

## Goal

Freeze the finite set of signature-only upgrades that are worth benchmarking against the measured baseline before any benchmark matrix, runtime sweep, or training run is defined.

## Scope

This step is limited to candidate selection.

It should:

- convert the review in `docs/signature/review.md` into a frozen upgrade set for this repository
- define each candidate as an exact delta from the baseline signature recipe in `docs/signature/plan/baseline.md`
- record which proposals are explicitly deferred or excluded from the first exploration pass
- identify the combined exploratory bundle that later steps may treat as a single candidate

It should not:

- run training, evaluation, or runtime benchmarks
- change the D3QN host branch settings outside the signature config subtree
- widen scope to algorithm-family comparisons, reward changes, action-space changes, or risk-control changes
- redesign the environment to support new account-history channels
- define the benchmark matrix, acceptance thresholds, or runtime budget policy for later steps

## Inputs

- `docs/signature/plan/baseline.md`
- `docs/signature/plan/baseline_metrics.md`
- `docs/signature/improvements.md`
- `docs/signature/review.md`
- `docs/signature/status.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/test_signature_explore.yaml`
- implementation references:
  - `rl/features/signature.py`
  - `rl/envs/wrappers.py`
  - `rl/envs/make_env.py`

## Deliverables

- one frozen candidate inventory for the first signature exploration pass
- exact config-level definitions for each included candidate relative to the frozen baseline
- one explicit deferred list for proposals that are not entering the first pass
- one explicit combined exploratory bundle definition that later steps can benchmark as a single package
- enough candidate IDs and definitions for Step 2 to build a benchmark matrix without reopening scope

## Notes

### Frozen baseline reference

All candidates in this step must be defined as deltas from the frozen baseline in `docs/signature/plan/baseline.md`:

- embedding = `log_price + log_return + rolling_vol(window=5)`
- `logsig.degree = 3`
- `logsig.time_aug = true`
- `logsig.lead_lag = false`
- `standardize_path_channels = false`
- `basepoint = false`
- `subwindow_sizes = []`

Non-signature settings must remain unchanged while candidate configs are created.

### Frozen candidate upgrade set

The first exploration pass is limited to upgrades that are already implemented and benchmarkable in the current repository.

| Candidate ID | Upgrade | Exact delta from baseline | Status in repo |
|---|---|---|---|
| `C1_std` | Per-channel standardization | set `standardize_path_channels: true` | Implemented |
| `C2_bp` | Basepoint augmentation | set `basepoint: true` | Implemented |
| `C3_volprof` | Normalized cumulative volume channel | add `normalized_cumulative_volume` to `embedding` | Implemented |
| `C4_hlrange` | High-low range channel | add `high_low_range` to `embedding` | Implemented |
| `C5_multi` | Multi-scale sub-window concatenation | set `subwindow_sizes: [8, 16, 24]` | Implemented |
| `C6_deg4` | Higher truncation depth sweep | change `logsig.degree` from `3` to `4` only | Implemented |

### Frozen combined exploratory bundle

The bundled feature candidate for later steps is:

- `B1_explore = C1_std + C2_bp + C3_volprof + C4_hlrange + C5_multi`

This bundle matches the feature-level direction already captured in `configs/test_signature_explore.yaml`, while keeping:

- `logsig.degree = 3`
- `logsig.time_aug = true`
- `logsig.lead_lag = false`
- the existing account-feature append path unchanged

`C6_deg4` stays separate from `B1_explore` at freeze time so later steps can decide whether depth 4 is tested:

- as a standalone delta on top of baseline
- on top of `B1_explore`
- or both, if the benchmark matrix remains tractable after runtime checks

### Explicitly deferred from the first pass

The following proposals are not part of the first candidate set:

- factorial rescaling of logsignature terms
- partial lead-lag path construction focused on price
- using full-path `lead_lag: true` as a substitute for the partial lead-lag proposal
- OHLC intrabar micro-path construction
- agent inventory or account history embedded as a live path channel

These are deferred because they require architectural work, additional theoretical validation, or new environment/history plumbing beyond a clean config-level sweep.

### Practical rule for later steps

- Step 2 may choose how to benchmark the frozen candidates, but it must not introduce new candidate ideas.
- Step 3 and later steps may drop a frozen candidate only for a documented operational reason such as excessive runtime, invalid observation shape, or clearly broken behavior.
- If a new idea appears after this step, it should be added only by explicitly revising this file and the board status together.

## Evidence

- Frozen candidate inventory: `docs/signature/plan/step1_candidates.md`
- `f3` screening batch results: `docs/signature/plan/step1_screening_f3_results.md`
- Candidate rationale and readiness review: `docs/signature/review.md`
- Current implementation coverage: `docs/signature/status.md`
- Frozen exploratory bundle reference: `configs/test_signature_explore.yaml`
- Materialized Step 1 config family: `configs/signature_step1/`
