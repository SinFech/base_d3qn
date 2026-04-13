# Step 13 Return-Vol Window Single-Seed Full Results

## Scope

This note records the `S13_single_seed_full` outcome from Step 13.

## Result

No Step 13 candidate-seed pair was promoted from the short screen.

Therefore:

- no new Step 13 single-seed full run was executed
- no new matched `DW5_return_vol5` full control was required

## Implication

The Step 13 conclusion is entirely determined by the short-screen gate:

- none of `DW3_return_vol3`, `DW10_return_vol10`, or `DW20_return_vol20` displaced `DW5_return_vol5`
- within the `log_return + rolling_vol(window=k)` family, `window=5` remains the strongest executable horizon

## Final Reporting Interpretation

Because no Step 13 candidate entered the full stage:

- Step 13 produced no new cherry-pick winner versus the official baseline `f1` `5`-seed average
- the best known evidence for the family still comes from the reused `DW5_return_vol5` / `D3_return_vol5` branch recorded in Step 11

## Source files

- `docs/signature/plan/step13.md`
- `docs/signature/plan/step13_window_short_results.md`
