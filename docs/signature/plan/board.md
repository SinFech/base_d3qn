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

## Current Focus

- Active step: `None`
- Immediate objective: Signature Step 6 is complete; baseline stays default and `C4_hlrange` remains an exploratory robustness branch
