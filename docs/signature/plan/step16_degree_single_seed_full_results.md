# Step 16 Degree Single-Seed Full Results

## Scope

This note records the `S16_single_seed_full` outcome from Step 16.

## Result

No Step 16 candidate-seed pair was promoted from the short screen.

Therefore:

- no new Step 16 single-seed full run was executed
- no new matched `L3_deg3` full control was required

## Implication

The Step 16 conclusion is entirely determined by the short-screen gate:

- `degree=3` remained the strongest balanced short-run setting
- `degree=2` and `degree=4` each showed partial one-metric spikes only
- `degree=1` was structurally infeasible under the frozen `ConvDuelingDQN` backbone

## Final Reporting Interpretation

Because no Step 16 candidate entered the full stage:

- Step 16 produced no new cherry-pick winner versus the official baseline `f1` `5`-seed average
- the repository still has no degree-only full-run candidate that beats the official baseline on both `f1` metrics

## Source files

- `docs/signature/plan/step16.md`
- `docs/signature/plan/step16_degree_short_results.md`
