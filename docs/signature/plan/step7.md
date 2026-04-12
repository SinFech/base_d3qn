# Step 7 - Sweep Rolling-Vol Window Length

## Goal

Measure whether changing the `rolling_vol` embedding horizon improves the frozen baseline signature recipe, and determine whether any alternative window is worth carrying beyond a short-run sweep.

## Scope

This step is limited to one local signature ablation:

- vary only `embedding.rolling_vol.window`
- keep the rest of the signature recipe identical to the frozen baseline
- compare every window candidate against the current default `window=5` control

It should:

- define one compact rolling-vol window sweep around the current default
- run a matched short-run comparison on the hardest fold
- keep at most one non-default window for an optional full evaluation
- record whether `window=5` should remain the default volatility horizon

It should not:

- combine the window sweep with new channels, basepoint, standardization, or multi-scale changes
- reopen the closed Step 1 candidates
- change the action space, PER, n-step, risk controls, reward, or training loop
- treat a short-run sweep as enough to change the repository default by itself

## Inputs

- `docs/signature/plan/baseline.md`
- `docs/signature/plan/baseline_metrics.md`
- `docs/signature/plan/step6.md`
- `docs/signature/plan/step6_recommendation.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/folds_signature_step1_screen_f3.json`
- `configs/folds_rolling_long_oos.json`
- `scripts/walk_forward_protocol.py`
- `rl/features/signature.py`

## Deliverables

- one frozen rolling-vol window sweep matrix
- one config family for the selected window candidates
- one short-run result record against the matched `window=5` control
- zero or one promoted window candidate for a full walk-forward comparison
- one explicit keep-or-drop decision for each tested window

## Notes

### Baseline control

The control for this step remains the frozen baseline recipe:

- config family: `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- embedding:
  - `log_price`
  - `log_return`
  - `rolling_vol(window=5)`

All comparisons in this step should be interpreted relative to that control.

### Frozen sweep candidates

Use one compact set of window lengths inside the existing `24`-bar observation window:

| Candidate ID | `rolling_vol.window` | Role | Rationale |
|---|---:|---|---|
| `RV3` | `3` | short-horizon candidate | test whether very local volatility helps the policy react faster |
| `RV5` | `5` | control | current shipped baseline |
| `RV10` | `10` | medium-horizon candidate | test whether more smoothing improves stability |
| `RV20` | `20` | long-horizon candidate | test near-full-window volatility context without exceeding the observation window |

Do not include:

- `window <= 1`
  - this degenerates `rolling_vol` into an all-zero channel in the current implementation
- `window > 24`
  - this is not useful for a `24`-bar observation window and weakens interpretability

### Config rule

Every Step 7 config should be a direct copy of the frozen baseline except for:

- `env.obs.signature.embedding.rolling_vol.window`

No other config differences are allowed in the Step 7 family.

### Runtime rule

A dedicated runtime gate is not required for this step because:

- observation dimension does not change when only the rolling-vol window changes
- the wrapper path shape is unchanged
- the computation remains the same rolling-standard-deviation path with a different horizon

If an implementation change later modifies the runtime path materially, this assumption must be revisited.

### Recommended execution protocol

#### `S7_short`

Start with the same cheap hardest-fold filter used earlier:

- fold protocol:
  - `configs/folds_signature_step1_screen_f3.json`
- fold:
  - `f3` only
- seeds:
  - `42,43,44`
- default short budget:
  - `train.num_episodes = 80`
  - `train.max_total_steps = 30000`
- control:
  - `RV5`

#### `S7_full`

Promote at most one non-default survivor to a full walk-forward comparison:

- fold protocol:
  - `configs/folds_rolling_long_oos.json`
- seeds:
  - `42,43,44,45,46`
- control:
  - official baseline full-run with `window=5`

If no non-default window clearly beats `RV5` in `S7_short`, stop after the short sweep.

### Acceptance rule

For the short sweep, reuse the same style of gate as Step 4:

- a non-default window is worth promoting only if it beats `RV5` on at least one primary `f3` metric:
  - higher `oos_sharpe_mean`, or
  - higher `oos_return_pct_mean`
- and it does not regress catastrophically on the other primary metric:
  - no worse than `0.05` Sharpe points, or
  - no worse than `5.0` return-percentage points

For the optional full run, the final default decision should remain strict:

- `window=5` should stay default unless the promoted window avoids material regression on overall OOS metrics
- one strong `f3` gain alone is not enough to change the repository default

### Expected outputs

If this step is executed, write results to:

- `docs/signature/plan/step7_window_sweep_results.md`
- optionally `docs/signature/plan/step7_full_results.md`
- `runs/step7_short_volwindow_f3/`
- optionally `runs/step7_full_volwindow_best/`

### Execution outcome

Step 7 is considered complete once the short sweep answers the step question:

- whether any non-default window is worth carrying beyond the short-run filter
- and which single window, if any, should be promoted

For this step, that threshold was met by the short sweep itself:

- `RV10` was the strongest non-default window
- `RV20` also beat the control, but ranked below `RV10`
- `RV3` underperformed the control

Therefore:

- `RV10` is the promoted follow-up candidate
- `window=5` remains the repository default until a later step explicitly runs and records a full official comparison
- no Step 7 default change is made from short-run evidence alone

## Evidence

- Current default baseline definition: `docs/signature/plan/baseline.md`
- Current default recommendation: `docs/signature/plan/step6_recommendation.md`
- Rolling-vol implementation: `rl/features/signature.py`
- Step 7 short-sweep result record: `docs/signature/plan/step7_window_sweep_results.md`
