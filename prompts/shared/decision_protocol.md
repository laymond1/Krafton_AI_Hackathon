# Shared Decision Protocol

Use this protocol when comparing agent outputs, resolving conflicts, or selecting a final direction.

## Core rule

The human controller makes the final decision. Agents support the decision process but do not replace it.

## Decision order

1. Confirm the exact objective and required submission form.
2. Eliminate any output that contains unsupported claims, misses explicit constraints, or ignores the stated deliverable.
3. Compare remaining options using the shared scoring rubric.
4. Merge the strongest components if a hybrid is better than any single output.
5. Record the selected direction, the reason, and the remaining risks.

## Conflict resolution rules

When outputs disagree:

1. Identify the exact conflicting claim.
2. Check whether the claim is a direct fact, a reasonable assumption, or an unsupported leap.
3. Prefer the interpretation with the strongest evidence from the problem statement.
4. If evidence is insufficient, preserve the ambiguity and treat it as an assumption or explicit risk.
5. Prefer the lower-risk path when the time cost of resolving ambiguity is too high.

## Tie-break rules

If two options appear similarly strong, prefer the one that:

- covers more explicit requirements
- relies on fewer assumptions
- is easier to execute and verify within the time limit
- is easier to explain clearly in a final submission
- leaves less chance of reviewer confusion

## Merge rules

Hybridization is allowed and often useful:

- take problem framing and risk analysis from the stronger reasoning output
- take execution steps and verification logic from the stronger operational output
- do not merge conflicting claims without resolving or labeling them

## Escalation triggers

Escalate to the human controller immediately if:

- a key requirement is ambiguous
- two outputs recommend materially different strategies with different risk profiles
- the final answer depends on an assumption that has not been approved
- the submission appears polished but unsupported

## Required decision note

Record a short decision note in this format:

```md
# Decision
- Selected direction:

# Why This Direction
- Evidence:
- Time advantage:
- Main tradeoff:

# Rejected Alternative
- Alternative:
- Reason not selected:

# Remaining Risks
- Risk 1:
- Risk 2:

# Next Action
- Owner:
- File to update:
- Deadline:
```
