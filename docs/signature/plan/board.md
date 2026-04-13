# Signature Exploration Board

This board tracks the execution status of the signature-component exploration plan.

See `AGENTS.md` in this directory for the operating rules of the planning workspace.

## Status Legend

- `Planned`: the step exists, but execution has not started yet.
- `In Progress`: the step is being worked on, including evidence gathering and step-note updates.
- `Blocked`: the step cannot move because a dependency, decision, or resource is missing.
- `Done`: the step goal has been fully executed and its deliverables are actually completed. Writing or polishing the step document alone is not enough.

## Board

| Step | Title | Status | Goal |
|---|---|---|---|
| [Step 0](./step0.md) | Measure Baseline Metrics | Done | Measure and record the baseline metrics that later signature candidates will be compared against. |
| [Step 1](./step1.md) | Freeze Candidate Signature Upgrade Set | Done | Lock the set of improvements that are worth testing in this repository. |
| [Step 2](./step2.md) | Define Benchmark Matrix and Acceptance Criteria | Done | Define which configs, metrics, and thresholds will be used for comparison. |
| [Step 3](./step3.md) | Measure Feature-Extraction Runtime and Observation Cost | Done | Benchmark observation dimension and signature extraction cost for each candidate setup. |
| [Step 4](./step4.md) | Run Short Training and Stability Sweeps | Done | Use short runs to filter weak candidates before full evaluation. |
| [Step 5](./step5.md) | Run Full Training and Evaluation on Final Candidates | Done | Compare the strongest candidates under the shared evaluation protocol. |
| [Step 6](./step6.md) | Write Adoption Recommendation and Default Decision | Done | Decide which signature options should remain exploratory and which should be promoted. |
| [Step 7](./step7.md) | Sweep Rolling-Vol Window Length | Done | Test whether alternative `rolling_vol.window` horizons outperform the current `window=5` baseline. |
| [Step 8](./step8.md) | Run Full Walk-Forward Comparison for `RV10` | Done | Determine whether the promoted `rolling_vol.window=10` candidate is strong enough to replace the current default. |
| [Step 9](./step9.md) | Run an `f1`-Only Specialist Search | Done | Search for a signature configuration that improves only the `f1` OOS Sharpe and OOS Return targets. |
| [Step 10](./step10.md) | Validate Single-Seed `f1` Cherry Picks | Done | Test whether the strongest Step 9 short-run single-seed wins survive under matched single-seed full runs. |
| [Step 11](./step11.md) | Search `f1` Cherry Picks via Embedding Reduction | Done | Test whether removing baseline embedding channels creates stronger `f1` cherry-pick behavior. |
| [Step 12](./step12.md) | Search `f1` Cherry Picks via Embedding Replacement | Done | Test whether one-for-one channel replacements outperform the current baseline embedding on `f1` cherry-pick criteria. |
| [Step 13](./step13.md) | Sweep `rolling_vol.window` on the `D5` Return Base | Done | Test which `rolling_vol.window` horizon works best when `log_return` is the only non-volatility path signal. |
| [Step 14](./step14.md) | Reopen Replacement Candidates Blocked by Scope Gates | Done | Re-test replacement-style embedding candidates that were previously excluded by scope rather than by a clear negative result. |
| [Step 15](./step15.md) | Validate the Encoder Bottleneck Hypothesis | Done | Test whether switching from `ConvDuelingDQN` to `MLPDuelingDQN` unlocks stronger `f1` performance for the main specialist signature candidates. |
| [Step 16](./step16.md) | Sweep `logsig.degree` on the Frozen Baseline Embedding | Done | Test whether lower or higher signature truncation levels outperform the current `degree=3` setting on the `f1` specialist objective. |

## Current Focus

- Active step: none
- Immediate objective: decide whether any follow-up should target a larger signature-capacity branch or stop the degree line here
