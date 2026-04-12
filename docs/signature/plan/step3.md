# Step 3 - Measure Feature-Extraction Runtime and Observation Cost

## Goal

Measure the runtime and observation-size cost of the active Step 2 shortlist before spending more training budget on short-run or full evaluation.

## Scope

This step is limited to the matched runtime benchmark defined in `step2.md`.

It should:

- benchmark the active shortlist only:
  - `baseline`
  - `C1_std`
  - `C4_hlrange`
- measure the full signature observation path through `SignatureObsWrapper`
- record `obs_dim`, extraction speed, and relative runtime cost vs the matched baseline control
- decide which candidates pass the `S3_runtime` gate and may continue to Step 4

It should not:

- introduce new candidates
- use a different observation path than training uses
- rank candidates by OOS return or Sharpe
- run the short training sweeps themselves

## Inputs

- `docs/signature/plan/step2.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/signature_step1/c1_std.yaml`
- `configs/signature_step1/c4_hlrange.yaml`
- `scripts/benchmark_signature_wrapper.py`
- `rl/envs/make_env.py`
- `rl/envs/wrappers.py`

## Deliverables

- one runtime measurement record for the Step 2 shortlist
- recorded values for each candidate:
  - `obs_dim`
  - `ms_per_window_mean`
  - `windows_per_sec_mean`
  - `obs_dim_ratio` vs baseline
  - `runtime_ratio` vs baseline
- one explicit pass/fail decision for the `S3_runtime` gate

## Notes

### Frozen runtime protocol

- Candidates:
  - `baseline`
  - `C1_std`
  - `C4_hlrange`
- Script:
  - `scripts/benchmark_signature_wrapper.py`
- Measurement path:
  - build env through `make_env(...)`
  - use the wrapped env `get_state()` path from `SignatureObsWrapper`
- Benchmark settings:
  - `num_windows = 512`
  - `warmup = 64`
  - `repeats = 5`
  - device override for benchmark script: `cpu`
- Output root:
  - `runs/step2_runtime/`

### Gate rule

A candidate passes Step 3 only if:

- env/signature construction succeeds
- `obs_dim_ratio <= 2.0x`
- `runtime_ratio <= 2.0x`

## Evidence

- Runtime measurement record: `docs/signature/plan/step3_runtime_results.md`
- Raw benchmark outputs: `runs/step2_runtime/`
