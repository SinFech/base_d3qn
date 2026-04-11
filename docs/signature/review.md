# Signature Improvements Review

_Reviewed on 2026-04-11._

This note reviews the proposals in `docs/signature/improvements.md` against the current repository architecture and marks which ideas are reasonable to explore now.

## Recommended now

These changes fit the current codebase, have a clear implementation path, and can be benchmarked without redesigning the environment stack.

1. Per-channel standardization before the signature
   - Reasonable for the current mixed-scale channel setup.
   - Implemented as `standardize_path_channels`.
   - Current implementation scales each channel so the standard deviation of first differences is approximately `1`, when that scale is well-defined.

2. Basepoint augmentation
   - Low-risk and local to the path builder.
   - Implemented as `basepoint`.
   - Useful once absolute level information matters, especially when the path contains price-level-derived channels.

3. Normalized cumulative volume channel
   - Reasonable because the dataset already contains `Vol.` and the loader now parses it into numeric `Volume`.
   - Implemented as embedding channel `normalized_cumulative_volume`.

4. High-low range channel
   - Reasonable because the dataset already contains `High` and `Low`.
   - Implemented as embedding channel `high_low_range`.
   - This is a practical substitute for spread-like information in the current daily-bar dataset.

5. Multi-scale sub-window concatenation
   - Reasonable and cheap to explore.
   - Implemented as `subwindow_sizes`.
   - Helps recover some temporal locality that a single whole-window signature discards.

## Recommended as a sweep, not a hard-coded default

1. Increase truncation depth to 4
   - Worth testing, but it should stay a config sweep rather than become the new default immediately.
   - Depth 4 increases feature dimension and runtime sharply, especially when combined with extra channels and multi-scale windows.
   - The new benchmark path should be used first to measure the cost.

## Deferred

These ideas may still be useful, but they need more architectural work or stronger theoretical alignment with the current implementation.

1. Factorial rescaling of terms
   - Deferred.
   - The current pipeline uses `log_sig`, not raw `sig`.
   - The argument in `improvements.md` applies most directly to raw signature levels, while logsignature coordinates depend on basis construction and are not a simple per-level tensor block.

2. Partial lead-lag transform focused on price
   - Deferred.
   - `pysiglib` exposes a full-path `lead_lag` flag, but the proposal is for a partial lead-lag transform on selected channels only.
   - That requires a custom path construction layer rather than a config toggle.

3. OHLC intrabar micro-path
   - Deferred.
   - Plausible, but it changes the meaning of each bar from a single point to a short intra-bar path and needs careful design and testing.

4. Inventory as a live path channel
   - Deferred.
   - The current environments expose current account state, but they do not retain an aligned history of account state across the observation window.
   - Broadcasting the current position across the whole window would not faithfully implement the proposal.

## Practical starting point

The exploratory config `configs/test_signature_explore.yaml` is the current recommended starting point for feature-level benchmarks. It combines:

- `log_price`
- `log_return`
- `rolling_vol`
- `normalized_cumulative_volume`
- `high_low_range`
- `standardize_path_channels: true`
- `basepoint: true`
- `subwindow_sizes: [8, 16, 24]`

This keeps the transform close to the shipped baseline while expanding the path information in ways that are both implementable and measurable.
