# Shared Scoring Rubric

Use this rubric for fast side-by-side comparison of Claude and Codex outputs.

Score each category from 1 to 5:

- `1` = weak or unsafe
- `3` = usable with edits
- `5` = strong and ready to build on

## Categories

| Category | What to measure | 1 | 3 | 5 |
| --- | --- | --- | --- | --- |
| Requirement coverage | Does it address the stated task and constraints? | misses important requirements | covers core requirements but leaves gaps | covers all visible requirements cleanly |
| Factual discipline | Does it separate facts and assumptions correctly? | mixes facts and guesses | mostly separated with minor leakage | rigorously separated and transparent |
| Practicality under time pressure | Can this be executed in a 4-hour hackathon window? | unrealistic or too heavy | feasible with focus | highly executable and time-aware |
| Clarity | Is it easy for the human controller to review quickly? | confusing or bloated | understandable with some effort | concise, structured, and high signal |
| Risk handling | Does it surface real failure modes and uncertainty? | ignores major risks | catches some important risks | exposes major risks and mitigations clearly |
| Submission readiness | How close is it to a final answer or usable draft? | far from usable | partially usable | close to submit-ready |

## Suggested interpretation

- `26-30`: strong candidate for adoption
- `20-25`: usable but needs edits
- `14-19`: only use selected parts
- `6-13`: too weak to rely on directly

## Fast comparison steps

1. Score Claude output.
2. Score Codex output.
3. Compare totals.
4. Check whether the lower-scoring output still has one superior component worth merging.
5. Record the decision note before moving to the final draft.

## Red flags

Reduce the score immediately if the output:

- states assumptions as facts
- ignores a hard constraint
- sounds polished but is not grounded in the problem
- creates more work to validate than it saves
