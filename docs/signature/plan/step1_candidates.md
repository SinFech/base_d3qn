# Step 1 Candidate Inventory

_Recorded on 2026-04-11 for Step 1._

## Decision

Step 1 freezes the first-pass signature candidate set as a small D3QN config family derived from the measured baseline:

- baseline config: `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- candidate config directory: `configs/signature_step1/`

All Step 1 candidate configs keep the same:

- dataset and date range
- environment reward and capital/risk settings
- action space and account-feature append path
- D3QN hyperparameters, PER settings, and `n_step`

Only the signature configuration is allowed to change in this family.

## Frozen candidate config family

| Candidate ID | Config file | Change from baseline | Purpose |
|---|---|---|---|
| `C1_std` | `configs/signature_step1/c1_std.yaml` | `standardize_path_channels: true` | Test mixed-scale channel normalization alone. |
| `C2_bp` | `configs/signature_step1/c2_bp.yaml` | `basepoint: true` | Test whether absolute path anchoring helps on its own. |
| `C3_volprof` | `configs/signature_step1/c3_volprof.yaml` | add `normalized_cumulative_volume` to `embedding` | Test whether volume-profile shape adds useful signal. |
| `C4_hlrange` | `configs/signature_step1/c4_hlrange.yaml` | add `high_low_range` to `embedding` | Test whether range/spread proxy adds useful signal. |
| `C5_multi` | `configs/signature_step1/c5_multi.yaml` | `subwindow_sizes: [8, 16, 24]` | Test temporal-locality recovery without changing channels. |
| `C6_deg4` | `configs/signature_step1/c6_deg4.yaml` | set `logsig.degree: 4` | Test depth 4 as a standalone runtime/performance sweep. |
| `B1_explore` | `configs/signature_step1/b1_explore.yaml` | combine `C1_std + C2_bp + C3_volprof + C4_hlrange + C5_multi` | Test the main exploratory feature bundle at depth 3. |

## Sanity-check results

The candidate configs were validated by loading each file through the D3QN config loader and constructing the wrapped environment successfully.

This is not a runtime benchmark. It is only a config-validity and observation-shape check.

| Config | Resolved `obs_dim` |
|---|---:|
| baseline `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml` | `34` |
| `configs/signature_step1/c1_std.yaml` | `34` |
| `configs/signature_step1/c2_bp.yaml` | `34` |
| `configs/signature_step1/c3_volprof.yaml` | `59` |
| `configs/signature_step1/c4_hlrange.yaml` | `59` |
| `configs/signature_step1/c5_multi.yaml` | `94` |
| `configs/signature_step1/c6_deg4.yaml` | `94` |
| `configs/signature_step1/b1_explore.yaml` | `277` |

## Bundle rule frozen in Step 1

`C6_deg4` is frozen as a separate candidate, not baked into `B1_explore`.

That means Step 2 may benchmark depth 4 in one of the following ways without reopening Step 1 scope:

- `C6_deg4` alone
- `B1_explore` alone
- `B1_explore` plus the `C6_deg4` delta, if later steps decide the combined variant is worth materializing after runtime checks

Step 1 does not freeze `B1_explore + degree4` as a separate required candidate.

## Deferred proposals

The following ideas are explicitly out of scope for the first-pass candidate family:

- factorial rescaling of logsignature terms
- partial lead-lag path construction focused on price
- enabling full-path `lead_lag: true` as a substitute for the partial lead-lag proposal
- OHLC intrabar micro-path construction
- embedding inventory or account history as a live path channel

These are deferred because they require architecture changes, stronger theoretical justification for the current logsignature basis, or new environment history plumbing.

## Execution note

No training or runtime benchmark was launched in Step 1.

Step 1 only freezes the candidate set and materializes exact config files so Step 2 can define a benchmark matrix without recreating candidate definitions by hand.
