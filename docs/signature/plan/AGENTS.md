# AGENTS.md

This file defines how to use the markdown files under `docs/signature/plan/`.

## Purpose

The `plan/` directory is the execution workspace for the signature exploration plan. It is not only a place to draft ideas. It is where step status, step evidence, and intermediate decisions are recorded while the work is being carried out.

## Source of truth

- `board.md` is the single source of truth for step status.
- Step status must describe execution progress, not document-writing progress.
- Do not mark a step as `Done` only because its step document was created or filled in.

## Status semantics

- `Planned`
  - The step has been defined, but work has not started.
- `In Progress`
  - The step is actively being executed.
  - This includes collecting evidence, refining the step note, creating supporting records, and updating decisions.
- `Blocked`
  - The step cannot proceed because of a missing prerequisite, unresolved decision, or missing data.
- `Done`
  - The step goal has been fully executed.
  - The required deliverables exist and the supporting evidence has been recorded in this directory or in linked repository artifacts.

## Expected workflow

1. Update `board.md` before or when starting a step.
2. Use the corresponding `stepN.md` file to define or refine:
   - goal
   - scope
   - inputs
   - deliverables
   - notes
3. Create additional markdown files in this directory when the step needs durable records such as:
   - benchmark notes
   - open questions
   - decision logs
   - measurements
   - result summaries
4. Link or reference those records from the relevant step document when useful.
5. Mark the step as `Done` in `board.md` only after the step has actually been executed and its deliverables are complete.

## Document usage rules

- Keep `board.md` concise. It should answer: what step is active, what is blocked, and what is done.
- Keep each `stepN.md` focused on the contract of the step, not raw logs.
- Put detailed evidence, measurements, or side investigations into separate markdown files when they would make the step file noisy.
- Prefer small, append-only updates over large rewrites so progress remains easy to audit.
- When a step changes scope, update both the step document and `board.md` in the same edit.

## Practical rule

If there is any doubt, keep the step at `In Progress` until a reviewer can see both:

- what was supposed to happen
- what actually happened

in the plan documents and linked artifacts.
