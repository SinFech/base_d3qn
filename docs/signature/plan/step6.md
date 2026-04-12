# Step 6 - Write Adoption Recommendation and Default Decision

## Goal

Turn the Step 5 evidence into one explicit repository decision:

- which signature recipe remains the default D3QN reference
- which surviving signature candidate, if any, should remain available as an exploratory branch
- which Step 1 candidates are closed unless scope is reopened later

## Scope

This step is limited to interpreting the already-frozen evidence from Steps 0-5.

It should:

- compare the official baseline and the surviving Step 5 candidate under the ranking order frozen in `step2.md`
- make one explicit default-decision for the repository
- make one explicit keep-or-close decision for `C4_hlrange`
- record what later work is still allowed without reopening the earlier screening process

It should not:

- run new experiments
- revise the benchmark matrix from `step2.md`
- reopen dropped Step 1 candidates by implication
- change the official baseline source in `baseline.md`

## Inputs

- `docs/signature/plan/baseline.md`
- `docs/signature/plan/baseline_metrics.md`
- `docs/signature/plan/step2.md`
- `docs/signature/plan/step3_runtime_results.md`
- `docs/signature/plan/step4_short_results.md`
- `docs/signature/plan/step5_full_results.md`
- `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- `configs/signature_step1/c4_hlrange.yaml`

## Deliverables

- one recorded adoption recommendation for the signature exploration pass
- one explicit default-config decision for the repository
- one explicit status for `C4_hlrange`
- one explicit closure status for the other Step 1 candidates
- one short list of allowed next-step directions if signature work continues

## Notes

### Decision target

Step 6 must answer a narrower question than the permissive Step 5 keep rule:

- not only whether `C4_hlrange` is worth keeping in scope
- but whether it is strong enough to replace the frozen baseline as the default signature recipe

### Default decision rule

The default recipe should change only if the Step 5 survivor is better enough to justify repository-wide promotion.

For this plan, promotion requires all of the following:

- no material regression on overall `oos_sharpe_mean`
- no material regression on overall `oos_return_pct_mean`
- no severe fold-level collapse that would make the candidate clearly harder to defend than the current baseline

This is stricter than the Step 5 keep rule because a default-config promotion affects later comparisons and user expectations.

### Step 6 outcome

The final recommendation is:

- keep `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml` as the default D3QN signature recipe
- keep `configs/signature_step1/c4_hlrange.yaml` as an exploratory robustness branch only
- close `C1_std`, `C2_bp`, `C3_volprof`, `C5_multi`, `C6_deg4`, and `B1_explore` for this pass

### Rationale summary

`C4_hlrange` is worth preserving because it improved the hardest fold under both the short-run gate and the full walk-forward evaluation.

It should not replace baseline because:

- aggregate OOS Sharpe regressed from `0.3132` to `0.2468`
- aggregate OOS Return regressed from `45.2441%` to `33.3150%`
- `f2` degraded sharply on both primary metrics

Its defensible value is narrower:

- better `f3` robustness
- slightly better `f1` return
- weaker aggregate upside than the baseline

### Allowed follow-up directions

If signature work continues after this step, the default scope should be:

- baseline-only reporting stays anchored on `configs/d3qn_signature_capital_6act_per06_n3_worstfold.yaml`
- any new signature follow-up should branch from `C4_hlrange` or a direct baseline-vs-`C4_hlrange` comparison
- dropped Step 1 candidates stay closed unless a later note explicitly reopens scope with a new reason

## Evidence

- Final decision record: `docs/signature/plan/step6_recommendation.md`
- Full comparison evidence: `docs/signature/plan/step5_full_results.md`
- Short-run survivor gate: `docs/signature/plan/step4_short_results.md`
- Runtime gate: `docs/signature/plan/step3_runtime_results.md`
