# Shared Output Format

Use this structure for all major agent outputs unless the human controller explicitly tells you not to.

## Hard rules

- Keep the section order unchanged.
- Separate facts from assumptions with zero ambiguity.
- If something is not stated in the problem, do not present it as a fact.
- If information is missing, say `Not stated in problem`.
- Write for decision speed, not for essay length.
- Prefer concise bullet points over long paragraphs.
- Optimize for submit-ready usefulness under a 4-hour hackathon constraint.
- Do not overclaim certainty.
- Do not replace human judgment.

## Required structure

```md
# Objective
- State the task in one or two lines.
- Describe the expected deliverable or submission form if known.

# Facts from Problem
- List only direct facts from the problem.
- Include explicit constraints, required outputs, judging signals, and format requirements.
- If helpful, quote short phrases from the problem exactly.

# Assumptions
- List inferred interpretations separately.
- Explain why each assumption matters.
- Mark unresolved ambiguity clearly.

# Proposed Approach
- Describe the recommended path forward.
- Keep it practical and time-aware.
- If multiple options matter, label them clearly.

# Risks
- List failure modes, weak assumptions, dependency risks, and likely reviewer objections.
- Put high-severity risks first.

# Deliverables
- State the concrete artifacts to produce next.
- Make them easy for the human controller to compare or approve.

# Immediate Next Actions
1. Give the next actions in priority order.
2. Keep actions short and executable.
3. Prefer actions that reduce uncertainty quickly.
```

## Section quality checks

- `Objective` should be specific enough that another agent can continue from it.
- `Facts from Problem` should survive a line-by-line challenge from the human controller.
- `Assumptions` should expose uncertainty, not hide it.
- `Proposed Approach` should fit the problem and the time limit.
- `Risks` should be decision-relevant, not generic.
- `Deliverables` should be concrete files, outputs, or submission artifacts.
- `Immediate Next Actions` should be executable without reinterpretation.

## Avoid

- mixing facts and assumptions
- vague claims such as `likely fine` or `should work` without a reason
- long background explanations that do not help a decision
- pretending the problem statement said more than it did
