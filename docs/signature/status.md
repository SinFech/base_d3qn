# Signature Implementation Status

_Last reviewed: 2026-04-11 on branch `sig-explore` after the exploratory feature pass._

This note summarizes the signature-related code that is already implemented in the repository today. It focuses on the actual shipped baseline, not the proposed upgrades listed in `docs/signature/improvements.md`.

## 1. Current implementation at a glance

- Signature observations are already integrated into the main training stack.
- The implementation is based on `pysiglib`, wrapped by repository code in `rl/features/signature.py`.
- Signature mode is exposed through `env.obs.type: signature` and can be used by D3QN, PPO, and SAC.
- The shipped baseline remains close-price-centric, but the signature stack now also supports exploratory OHLCV-derived channels and multi-scale concatenation through config.
- The active design is still a flat logsignature feature vector, optionally followed by extra scalar account features.

## 2. Core code paths

- `rl/features/signature.py`
  - `PathBuilder` converts a 1D close-price window into a multichannel path tensor.
  - `LogSigTransformer` calls `pysiglib.log_sig(...)` and exposes `obs_dim(...)`.
- `rl/envs/wrappers.py`
  - `SignatureObsWrapper` replaces the raw environment observation with the computed logsignature vector.
  - It can append extra scalar account features after the logsignature.
- `rl/envs/make_env.py`
  - When `obs.type == "signature"`, the env is wrapped automatically.
  - Only the `pysiglib` backend is supported at the moment.
- Trainer/config wiring
  - `rl/algos/d3qn/trainer.py`
  - `rl/algos/ppo/trainer.py`
  - `rl/algos/sac/trainer.py`
  - All three trainers parse the same `env.obs.signature` config subtree and build models from the resolved wrapped observation dimension.

## 3. What the current path builder actually does

- Input source
  - The wrapper can still consume `env.get_state()`, but when the environment exposes `data`, `t`, and `window_size`, it now slices the full market window directly.
  - This allows the path builder to use parsed OHLCV columns when they exist in the dataset.
- Supported embedding channels
  - `log_price`
  - `log_return`
  - `rolling_mean`
  - `rolling_vol`
  - `normalized_cumulative_volume`
  - `high_low_range`
- Alias normalization exists, so names like `price`, `return`, `vol`, and `rolling_std` map to the internal channel names.
- `log_price` is implemented as relative log-price, anchored to the first bar in the window.
- `log_return` is the first difference of log-price, with the first entry set to zero.
- `rolling_mean` and `rolling_vol` are computed from the log-return stream over a configurable rolling window.
- Optional path preprocessing now includes:
  - per-channel scaling via `standardize_path_channels`
  - explicit zero-vector prepending via `basepoint`

## 4. What the current logsignature transformer supports

- Backend: `pysiglib`
- Configurable parameters:
  - truncation degree
  - `method`
  - `time_aug`
  - `lead_lag`
  - `end_time`
  - torch device
  - torch dtype
  - `n_jobs`
  - optional `prepare_log_sig(...)` precomputation
- The observation dimension is not hard-coded. It is resolved at runtime from `pysiglib.log_sig_length(...)`.
- `SignatureObsWrapper` can now concatenate multiple logsignature vectors from configured suffix windows via `subwindow_sizes`.

## 5. How it is used in configs today

- Signature configs already exist for all three algorithm families:
  - `configs/d3qn_signature_capital.yaml`
  - `configs/ppo_signature.yaml`
  - `configs/sac_signature.yaml`
  - plus several D3QN walk-forward variants under `configs/d3qn_signature_capital_*`
- The common active pattern is:
  - embedding = `log_price + log_return + rolling_vol(window=5)`
  - logsignature degree = `3`
  - `time_aug = true`
  - `lead_lag = false`
- An exploratory config now exists at `configs/test_signature_explore.yaml` with:
  - volume and high-low channels
  - per-channel standardization
  - basepoint augmentation
  - multi-scale sub-window concatenation
- The older `configs/signature.yaml` and `configs/signature-3.yaml` also show earlier D3QN signature setups.
- The project metadata already treats signature support as a first-class dependency:
  - `pyproject.toml` depends on `pysiglib`
  - `uv.lock` includes a resolved `pysiglib` package

## 6. Account-state integration

- In capital-aware environments, extra account features can be appended after the logsignature.
- Current capital configs typically append:
  - `cash_ratio`
  - `position_ratio`
  - `equity_return`
  - `last_action`
- The discrete-capital and continuous environments expose more account fields than the current configs use.
- Important limitation: account state is appended after the signature. It is not embedded into the path before the signature transform.

## 7. Runtime and workflow coverage

- D3QN, PPO, and SAC all resolve the wrapped observation dimension dynamically, so the signature vector is usable in training and evaluation without hard-coding the input size.
- Default scripts already point to signature configs for PPO and SAC:
  - `scripts/train_ppo.py`
  - `scripts/eval_ppo.py`
  - `scripts/train_sac.py`
  - `scripts/eval_sac.py`
- Walk-forward and reporting scripts also reference signature configs:
  - `scripts/walk_forward_protocol.py`
  - `scripts/backtest_report.py`
- There is a smoke path for signature mode through `scripts/smoke_test.py` and `configs/test_signature.yaml`.

## 8. Concrete implementation caveats

- The original baseline remains close-only by default.
  - However, the current code can now consume parsed `Open`, `High`, `Low`, and `Volume` columns when a signature config requests them.
- Only `pysiglib` is supported.
  - `make_env(...)` raises an error for any other signature backend.
- `SignatureObsWrapper` has an `add_position` option, but the env factory currently hard-codes it to `False`.
- `prepare_cache_dir` is created and threaded through config objects, but the path is not passed into `pysiglib.prepare_log_sig(...)` in the current wrapper code.
- `make_env(...)` falls back to `"price_return"` when `signature.embedding` is missing, but `PathBuilder` only accepts a dict. In normal trainer usage this is avoided because the trainer config defaults already provide a dict embedding.

## 9. What is not implemented yet relative to `docs/signature/improvements.md`

The following ideas are still not part of the default shipped baseline:

- depth-4 as the default operating point
- factorial rescaling of higher-order terms
- partial lead-lag construction focused on price variance
- OHLC intrabar micro-path construction
- inventory/account variables embedded inside the path itself

The following ideas are now implemented as exploratory options rather than defaults:

- per-channel standardization before the signature
- explicit basepoint augmentation
- normalized cumulative volume as a path channel
- high-low range or spread-proxy channels
- multi-scale sub-window signature concatenation

## 10. Bottom line

The repository already has a real, end-to-end signature observation pipeline, and it is wired into D3QN, PPO, and SAC configs. The shipped baseline is still conservative, but the code now exposes a practical exploration layer on top of it: optional channel standardization, basepoint augmentation, parsed OHLCV-derived channels, and multi-scale concatenation, all behind config switches. Account variables are still appended after the signature rather than modeled jointly inside the path.
