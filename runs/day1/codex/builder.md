# Objective
- Turn the parsed Day 1 multiplication task into an execution-ready build plan for both mandatory subproblems.
- Produce low-friction artifacts that let the controller continue immediately with training, verification, and final packaging.

# Facts from Problem
- Two subproblems are mandatory: `Problem 1-1` exact hand-coded transformer weights with a correctness proof, and `Problem 1-2` a trainable transformer architecture that reaches `>=99%` accuracy.
- Inputs are fixed-position `A0..A5 B0..B5` tokens, zero-padded 6-bit binaries in LSB-first order.
- Outputs are fixed-position `P0..P11` tokens, zero-padded 12-bit binaries in LSB-first order.
- Full sequence length is 24 tokens, with autoregressive greedy decoding for the 12 output bits.
- All code must be Python + PyTorch, include at least one self-attention layer, and keep multiplication logic in weights rather than `if/else` code.
- Submission requires three scalar fields `P_1`, `P_2`, `Acc_2`, plus one ZIP containing a max-2-page PDF report and one Python file.
- Parameter counts must use unique parameters after tying/deduplication; fixed positional encodings do not count.

# Assumptions
- Accuracy should be treated as exact-match product accuracy, not per-bit accuracy.
- Why it matters: a small model that looks strong on per-bit accuracy may still fail the intended bar.
- `Problem 1-1` and `Problem 1-2` can use different models inside the same final Python file.
- Why it matters: the exact-proof track and the trainable-architecture track have different risk profiles and should not block each other.
- The safest cutoff policy is to preserve `P_1 = -1` unless the team has both an exact model and a proof that can survive review.
- Why it matters: unsupported exactness claims are likely worse than an explicit miss.
- The final single Python file may contain helper classes, training utilities, encoding helpers, and both problem entry points.
- Why it matters: Builder can prepare one reproducible draft file now instead of forcing a late merge.

# Proposed Approach
- Stage 1. Lock the contract first.
- Implement and verify reusable helpers for:
- 6-bit LSB-first input encoding,
- 12-bit LSB-first product encoding,
- autoregressive greedy decoding,
- unique-parameter counting.
- Output of this stage: a single-file scaffold that all later experiments use.
- Stage 2. Push `Problem 1-2` first because it is the higher-confidence path to a score.
- Build a minimal decoder-only transformer with an `AdderBoard`-inspired tiny Qwen-style block:
- tied binary embedding / output head,
- RoPE instead of learned positional embeddings,
- RMSNorm + causal self-attention,
- small SwiGLU-style MLP.
- Keep the default candidate small and configurable so nearby sizes can be swept quickly.
- Stage 3. Attach verification hooks before any real sweep.
- Add smoke tests for token-format correctness and shape checks.
- Add worked-example checks for `23 x 37 = 851` and `63 x 63 = 3969`.
- Add exhaustive-pair and sampled-pair evaluation helpers so verifier output is comparable.
- Stage 4. Timebox `Problem 1-1` as a separate proof track.
- Preferred path: convert binary long multiplication into explicit transformer responsibilities:
- head group for partial-product extraction,
- MLP group for carry aggregation,
- output readout for `P0..P11`.
- Cutoff path: if the proof is still incomplete after the first `Problem 1-2` sweep, freeze `1-1` as a documented open risk and continue toward a strong `1-2` package.
- Stage 5. Keep packaging in the loop.
- Log parameter counts, accuracy, and failed settings in the merged experiment log.
- Keep the report outline synchronized with actual evidence so the PDF is not reconstructed from memory.

# Risks
- High: `Problem 1-1` exact multiplication may still be infeasible to prove cleanly within the hackathon window even if a design sketch exists.
- High: the local environment currently lacks `torch`, so Builder can write the execution scaffold but cannot run training or inference smoke tests yet.
- Medium: a mismatch between exact-match accuracy and per-bit accuracy would distort model-size decisions if not enforced in the verifier.
- Medium: LSB-first ordering errors can silently invalidate otherwise correct models and proofs.
- Medium: a compact `Problem 1-2` model may need more than one sweep because the protocol fixes optimizer, epochs, and batch size.

# Deliverables
- Build Plan
- `runs/day1/codex/builder.md`
- Ordered execution stages with dependency order, fallback path, and cutoff policy.
- Files or artifacts to produce
- `runs/day1/codex/submission_draft.py`
- Single-file draft containing encoding helpers, model scaffold, training/eval hooks, greedy decoding, and parameter counting.
- `runs/day1/merged/solution_brief.md`
- Current selected direction centered on a low-risk `Problem 1-2` path and a timeboxed `Problem 1-1` proof track.
- Verification hooks
- `verify_worked_examples()` for token-order sanity.
- `unique_parameter_count()` for submission-ready model size reporting.
- `greedy_decode_product()` and `exact_match_accuracy()` for judge-like evaluation.
- `candidate_configs()` for small architecture sweeps.

# Immediate Next Actions
1. Install or activate a PyTorch-capable environment, then run the scaffold smoke tests first.
2. Run a small `Problem 1-2` candidate sweep around the default configuration and log `P_2` plus exact-match accuracy.
3. Decide a hard cutoff time for `Problem 1-1`; continue only if the proof can be written layer by layer, otherwise preserve `P_1 = -1`.
4. Update the merged experiment log and solution brief after each model-size decision.
5. Hand the scaffold to Verifier only after the exact same encode/decode path is used for training and evaluation.
