# Signature Benchmark Baseline

This note freezes the baseline reference and comparison rule used by the signature exploration plan.

## Baseline source

The baseline reference for this plan is the older `6-action, PER=0.6, n=3` D3QN walk-forward run group:

- run group: `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`
- config family: `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`

This baseline source is used because the user explicitly chose the strong `f1` result from this run group as the reference point for later signature experiments.

## Signature recipe tied to the baseline source

The baseline source uses the conservative shipped signature recipe:

- `embedding = {log_price, log_return, rolling_vol(window=5)}`
- `logsig.degree = 3`
- `logsig.time_aug = true`
- `logsig.lead_lag = false`
- `standardize_path_channels = false`
- `basepoint = false`
- `subwindow_sizes = []`

The signature exploration plan should still keep non-signature changes out of scope unless a later step explicitly widens scope.

That means:

- no action-space changes inside a candidate sweep
- no PER or n-step changes inside a candidate sweep
- no risk-control changes inside a candidate sweep
- no reward-definition changes
- no algorithm-family changes

## Baseline targets

The chosen run group defines four comparison targets:

1. `f1`
2. `f2`
3. `f3`
4. overall OOS aggregate across all folds

The user-selected headline baseline is the `f1` result:

- `f1 oos_sharpe_mean = 0.8904662600702572`
- `f1 oos_return_pct_mean = 85.86166745250878`

But later experiments must not evaluate only on `f1`.

Every candidate that reaches the full evaluation stage must report:

- `f1`
- `f2`
- `f3`
- overall OOS aggregate

## Metrics

The primary comparison metrics are:

- `oos_sharpe_mean`
- `oos_return_pct_mean`

The secondary diagnostic metrics are:

- `oos_reward_mean`
- `oos_sharpe_std`
- `oos_return_pct_std`
- `is_sharpe_mean`
- `is_sharpe_std`

For fold-level tables, use the fold-level means from `summary_by_algo_fold.csv`.

For overall OOS, use the aggregate values from `summary_by_algo.csv`.

## Acceptance rule

The comparison rule for this plan is intentionally permissive.

A candidate is considered worth keeping if at least one comparison target among:

- `f1`
- `f2`
- `f3`
- overall OOS

beats the corresponding baseline target on at least one primary metric:

- higher `oos_sharpe_mean`, or
- higher `oos_return_pct_mean`

This rule should be treated as a screening gate for exploration, not as a final deployment-quality robustness standard.

## Expected source files

When baseline or candidate results are read from retained walk-forward artifacts, the expected files are:

- `summary_by_algo_fold.csv`
- `summary_by_algo.csv`

The key fields used by this plan are:

- fold-level:
  - `fold_id`
  - `oos_sharpe_mean`
  - `oos_return_pct_mean`
  - `oos_reward_mean`
- aggregate:
  - `oos_sharpe_mean`
  - `oos_return_pct_mean`
  - `oos_reward_mean`
  - `oos_sharpe_std`
  - `oos_return_pct_std`
  - `is_sharpe_mean`
  - `is_sharpe_std`

## Rationale

- `reports/experiments.md` records the exact per-fold and aggregate values for `runs/wf_rolling_long_oos_d3qn_6act_a06_n3_worstfold`.
- The user explicitly chose the strong `f1` result from that run group as the baseline reference point.
- The user also explicitly required later experiments to report `f1`, `f2`, `f3`, and overall OOS rather than evaluating only on `f1`.
- Keeping the acceptance rule permissive is acceptable for early signature screening, as long as the full target table is still recorded.
