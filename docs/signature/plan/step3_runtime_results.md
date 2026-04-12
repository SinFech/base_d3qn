# Step 3 Runtime Results

_Recorded on 2026-04-11._

## Protocol

- Candidates:
  - `baseline`
  - `C1_std`
  - `C4_hlrange`
- Script:
  - `scripts/benchmark_signature_wrapper.py`
- Measurement path:
  - `make_env(...)`
  - wrapped env `get_state()` via `SignatureObsWrapper`
- Settings:
  - `num_windows = 512`
  - `warmup = 64`
  - `repeats = 5`
  - device = `cpu`
- Output root:
  - `runs/step2_runtime/`

## Results

| Candidate | `obs_dim` | `obs_dim_ratio` vs baseline | `ms_per_window_mean` | `runtime_ratio` vs baseline | `windows_per_sec_mean` | S3 gate |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | `34` | `1.0000` | `1.1173` | `1.0000` | `895.01` | Pass |
| `C1_std` | `34` | `1.0000` | `1.2377` | `1.1078` | `807.97` | Pass |
| `C4_hlrange` | `59` | `1.7353` | `1.1368` | `1.0175` | `879.70` | Pass |

## Interpretation

- `C1_std` preserved the same final observation size as baseline and added only a small runtime cost.
- `C4_hlrange` increased `obs_dim` to `59`, but stayed well under the `2.0x` observation-size gate.
- Both candidates also stayed well under the `2.0x` runtime gate.
- No Step 2 shortlist candidate was blocked by runtime or observation-cost concerns.

## Step 3 conclusion

The Step 3 survivors are:

- `baseline`
- `C1_std`
- `C4_hlrange`

All three continue to Step 4.

## Source files

- `runs/step2_runtime/baseline.txt`
- `runs/step2_runtime/c1_std.txt`
- `runs/step2_runtime/c4_hlrange.txt`
