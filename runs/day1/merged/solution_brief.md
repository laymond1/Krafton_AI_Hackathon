# Solution Brief

## Metadata

- Day: day1
- Solution owner: Codex Builder
- Based on: `runs/day1/raw_problem/problem.md`, `runs/day1/codex/parser.md`, `runs/day1/claude/parser.md`, `runs/day1/claude/strategist.md`, `runs/day1/claude/critic.md`
- Version: v0.2
- Status: Claude independent pass partially completed, then transferred to Codex due to session-usage limit; Codex now owns integrated execution path

# Objective

- Build a low-risk Day 1 path that can reach a defensible `Problem 1-2` result quickly while preserving a timeboxed attempt on `Problem 1-1`.
- Optimize for submit-ready execution speed, clear verifier handoff, and minimal late-stage packaging risk.

# Facts from Problem

- Both `Problem 1-1` and `Problem 1-2` are mandatory in the prompt.
- The model must be autoregressive, include at least one self-attention layer, and be implemented in Python + PyTorch.
- Input format is fixed 12-token LSB-first binary for `A0..A5 B0..B5`; output format is fixed 12-token LSB-first binary for `P0..P11`.
- The final submission must report `P_1`, `P_2`, and `Acc_2`, and include one PDF report plus one Python file.
- For `Problem 1-2`, the fixed training protocol uses AdamW, cosine annealing, batch size 256, 200 epochs, and 100,000 random training pairs.

# Assumptions

- Accuracy should be treated as exact-match product accuracy on sampled pairs.
- `Problem 1-1` and `Problem 1-2` may use different models inside the same final Python file.
- If the exact proof remains incomplete past the first successful `Problem 1-2` sweep, the team should prefer a truthful fallback over an unsupported `Problem 1-1` claim.

# Proposed Approach

- Prioritize a single-file execution scaffold for `Problem 1-2`: token encoding, decoder-only transformer, greedy decode, exact-match evaluation, and parameter counting.
- Use an `AdderBoard`-inspired tiny Qwen-style baseline as the main search path: tied embedding/output, RoPE, RMSNorm, and a very small SwiGLU MLP.
- Incorporate Claude strategist and critic findings into the Codex path instead of restarting analysis:
- keep `Problem 1-1` on a strict cutoff because no proof-backed exact design exists yet,
- treat tiny-sub-200-param results as diagnostic rather than submission-ready,
- preserve the warning that current empirical evidence is still far from the fixed-protocol `>=99%` target.
- Keep `Problem 1-1` on a separate proof track with an explicit cutoff. The build path should not stall on an exact hand-coded design unless the proof becomes layer-by-layer concrete.
- This approach is preferable under the 4-hour limit because it maximizes usable artifacts early and keeps packaging synchronized with evidence.

# Risks

- Local `.venv` runtime has `torch`, but it warns that `numpy` is not installed.
- `Problem 1-1` may remain unprovable within the time budget even if a rough design exists.
- Current `Problem 1-2` candidates learn bit structure faster than the original baseline, but exact-match remains far below the target.
- Claude critic raises an unresolved high-risk warning: the current tiny-family experiments are useful for diagnosis, but none are evidence for a final fixed-protocol score yet.
- LSB-first ordering mistakes can silently invalidate all later results.

# Deliverables

- Selected approach summary centered on `Problem 1-2` first and `Problem 1-1` timeboxed.
- Draft single-file scaffold at `runs/day1/codex/submission_draft.py`, now smoke-tested in `.venv`.
- Validation points:
- worked-example token checks,
- unique parameter counting,
- greedy exact-match evaluation,
- candidate configuration sweep hooks.
- Current observed baseline:
- original generic baseline was `100226` params and stayed around `0.0195` exact-match on short runs,
- new `AdderBoard`-inspired default is `182` params,
- 1-epoch micro-sweep reached up to `0.0469` exact-match and `0.5794` bit accuracy,
- 10-epoch run with 100,000 sampled pairs reached about `0.0254` exact-match and `0.6273` bit accuracy.
- bit-level analysis now shows why:
- candidate 3 (`51` params) collapses to predicting almost all output bits as `0`,
- candidate 5 (`142` params) predicts some `1`s but still underproduces them across most positions, especially mid bits `P02` to `P08`.
- candidate 6 (`158` params, untied output head + unshared norm) is the best current fix for that failure mode:
- it raises bit accuracy to about `0.638`,
- increases `pred1` across most positions relative to candidate 5,
- but still underpredicts `1`s in the low/mid band `P02` to `P08`, so exact-match remains around `0.032`.

# Immediate Next Actions

1. Promote candidate 6 as the current `Problem 1-2` control because it reduces underprediction without changing the training protocol.
2. Target the remaining low/mid-bit one-rate gap (`P02` to `P08`) rather than adding more generic width.
3. Keep Claude's strategist/critic output as reference only; route all further implementation and verification through Codex to avoid split ownership.
