# Objective
- Distill the top-5 `AdderBoard` leaderboard ideas from both Hand-Coded and Trained tracks into reusable design moves for Day 1 multiplication.
- Record which ideas were actually applied to `runs/day1/codex/submission_draft.py`.

# Facts from Problem
- Day 1 multiplication requires an autoregressive transformer with at least one self-attention layer.
- `Problem 1-2` asks for the smallest trainable architecture that can reach `>=99%` accuracy.
- `AdderBoard` is not the same task, but it is a strong reference for tiny arithmetic-transformer design patterns.

# Assumptions
- Addition leaderboard ideas that primarily compress alignment, normalization, and output readout can transfer to binary multiplication even if the carry logic is harder.
- Not every hand-coded trick is directly trainable; the best use is to convert them into trainable architectural biases rather than exact analytic weights.

# Proposed Approach
- Hand-Coded top-5 ideas extracted from `AdderBoard/README.md`:
- Rank 1: 1L Qwen-derived decoder, fixed Q, folded norm, tied carry hinge, shared carry scalar.
- Rank 2: phase-tied Q projection, coupled quadratic embedding, tied carry hinge, shared carry scalar.
- Rank 3: parametric tied embedding, gate tying via algebraic identity, merged carry scalar.
- Rank 4: sparse attention/MLP with constructive carry hinge.
- Rank 5: quadratic tied embedding, tied output head, RoPE digit routing, two-hinge ReLU MLP, parameterless pre-norm.
- Trained top-5 ideas extracted from `AdderBoard/README.md`:
- Rank 1: 1L Qwen3, d=3, hd=4, circular arc embedding, `K=rotation(Q)`, `V=Q`, tied `O=Q^T`, shared RMSNorms, tied QK norms.
- Rank 2: circular arc embedding, tied `K=V`, tied Q/O readout, shared RMSNorms, repeat-mix shared block.
- Rank 3: circular arc embedding, `K=rotation(Q)`, `V=Q`, tied `O=Q^T`, shared RMSNorms.
- Rank 4: circular arc embedding, `K=Q`, `V=Q`, tied `O=Q^T`, shared RMSNorms.
- Rank 5: circular arc embedding, `K=rotation(Q)`, `V=Q`, tied `O=Q^T`, shared RMSNorms.
- Applied to the multiplier scaffold:
- `quadratic` embedding style for hand-coded-inspired candidates.
- `arc` embedding style for trained-inspired candidates.
- `phase_tied` attention style to mimic phase-tied Q.
- `k_rot_q` and `k_eq_v` attention styles to mimic trained leaderboard tying patterns.
- `tie_o_to_q=True` to mimic tied attention output readout.
- `share_norm=True` to reuse one RMSNorm family.
- `activation='relu2'` for hand-coded sparse carry-style MLPs.
- `activation='swiglu'` for trained Qwen-style small models.

# Risks
- Multiplication may need more explicit multi-bit interaction than addition, so the smallest addition motifs may underfit exact-match even if bitwise accuracy improves.
- RoPE and tying patterns can improve parameter efficiency without solving the product-bit composition bottleneck.
- Hand-coded carry-hinge ideas may not be discoverable by SGD without better initialization or curriculum.

# Deliverables
- Reference map from leaderboard ideas to current candidate families.
- Candidate families now present in `runs/day1/codex/submission_draft.py`.
- Early-result interpretation:
- candidate 0: hand-inspired minimal phase-tied family, `28` params.
- candidate 3: trained-inspired arc + `K=V` + tied `O=Q^T`, `51` params.
- candidate 5: larger learned-embed control with shared/tied attention patterns, `142` params.

# Immediate Next Actions
1. Use candidate 3 as the main tiny trained baseline because it has the best current size-to-signal tradeoff.
2. Use candidate 0 as the smallest hand-inspired trainable reference, not yet as a final direction.
3. Add per-bit error analysis before making the next architecture jump.
