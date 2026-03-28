# Objective
- Build a submit-ready plan for Day 1: solve both required tasks for 6-bit binary multiplication with the smallest transformer(s) possible.
- Required submission shape from the problem: report `P_1`, `P_2`, `Acc_2`, plus one ZIP containing a max-2-page PDF report and one Python file.

# Facts from Problem
- Round context shown in the statement: KRAFTON AI R&D Hackathon, Round 1, Day 1, 4 hours.
- Core task: given `a, b ∈ {0, ..., 63}`, compute `a × b` with a transformer model.
- There are two mandatory subproblems:
- `Problem 1-1`: hand-coded transformer weights that perform exact multiplication, with as few parameters as possible, plus a correctness proof.
- `Problem 1-2`: a transformer architecture that can be trained to `>=99%` accuracy on multiplication, with as few parameters as possible.
- Evaluation signal stated for the task: estimate test accuracy by randomly sampling 10,000 pairs.
- Decoding rule: greedy decoding (`argmax`) at every output position.
- Fixed token format:
- Vocabulary is exactly two tokens: `0` and `1`, with token IDs `0` and `1`.
- Input tokens are `A0 A1 A2 A3 A4 A5 B0 B1 B2 B3 B4 B5` for 12 total tokens.
- Inputs are zero-padded 6-bit binaries in LSB-first order, with fixed positions and no separators.
- Output tokens are `P0 ... P11` for 12 total tokens.
- Outputs are zero-padded 12-bit binaries in LSB-first order.
- Full sequence length is 24 tokens total.
- The model generates `P0 ... P11` autoregressively conditioned on the 12-token input.
- Architecture rules marked as mandatory:
- All code must be Python + PyTorch.
- The model must contain at least one self-attention layer.
- The model must be autoregressive.
- `forward()` must take token ID tensors and return logits.
- No `if/else`, lookup tables, or control flow that encodes multiplication logic.
- Problem-specific knowledge must live in weights, not in code.
- Allowed choices explicitly include any positional encoding, any activation, weight tying, parameter sharing, low-rank/factorized projections, and custom embedding strategies for the 2-token vocabulary.
- Parameter counting rules:
- Count unique parameters after tying or deduplication.
- Fixed positional encodings do not count.
- Learned positional encodings do count.
- Bias terms count.
- All `nn.Parameter` values count.
- If `requires_grad=True` would apply in training, it counts.
- `Problem 1-1` requires a written layer-by-layer correctness argument:
- Explain what each attention head attends to.
- Explain what each MLP computes.
- Explain how outputs compose into final product bits.
- “A numerical dump of weights without explanation will receive no credit.”
- `Problem 1-2` submission is only the architecture: a `build_model()` function returning an untrained `nn.Module`.
- Fixed training protocol for `Problem 1-2`:
- Initialization: PyTorch defaults.
- Training data: 100,000 random pairs with `a, b ∈ [0, 63]`.
- Optimizer: `AdamW(lr=1e-3, weight_decay=0.01)`.
- Schedule: cosine annealing over 200 epochs.
- Batch size: 256.
- Loss: cross-entropy on the 12 output-token positions only.
- Required scalar submission fields:
- `P_1`: parameter count for Problem 1-1, or `-1` if no solution found.
- `P_2`: parameter count for Problem 1-2.
- `Acc_2`: accuracy for Problem 1-2 in `[0.0, 1.0]`.
- Required ZIP contents:
- One PDF report, max 2 pages.
- The report must cover architecture description, Problem 1-1 approach and correctness proof, Problem 1-2 approach, training curve, accuracy vs. model size, and ablations / failed attempts.
- One single Python file that defines the model and reproduces the results.
- Explicitly required versus optional design freedom:
- Required: PyTorch, self-attention, autoregressive generation, standard `forward()` returning logits, the fixed token format, the fixed `Problem 1-2` training protocol, the three scalar submission fields, the PDF report, and the single Python file.
- Optional / design choice: positional encoding type, activation function, weight tying, parameter sharing, low-rank factorization, custom embedding strategy, and the specific architecture chosen for `Problem 1-2`.

# Assumptions
- Accuracy definition is not stated in problem.
- Assumption: use exact-match product accuracy over sampled input pairs, not per-bit accuracy.
- Why it matters: model-size decisions and verifier thresholds change substantially depending on this interpretation.
- Inference procedure for autoregressive decoding is not stated in problem beyond “greedy decoding.”
- Assumption: the 12 input bits are provided as prompt context, then output bits are generated one by one with a causal mask.
- Why it matters: builder and verifier should test the same decode path the judges are likely to use.
- The problem does not state that both subproblems must share one architecture.
- Assumption: `Problem 1-1` and `Problem 1-2` may use different models if that reduces risk or parameter count.
- Why it matters: execution can split into a proof-oriented exact design for `1-1` and a trainable compact design for `1-2`.
- The statement says “single Python file” but does not state whether helper functions/classes inside that file are acceptable.
- Assumption: one `.py` file may contain multiple classes/functions as long as it fully reproduces the submission.
- Why it matters: packaging can stay simple without forcing unnatural inlining later.

# Proposed Approach
- Treat the work as two separate tracks with one shared encoding/verification harness.
- First lock the exact I/O contract:
- Implement reusable encode/decode helpers for 6-bit input and 12-bit output in LSB-first order.
- Add a verifier that checks token ordering, greedy decoding behavior, and parameter counting rules before any model comparison.
- For `Problem 1-1`, optimize for proofability before raw compactness.
- Start from binary long multiplication: partial products plus carry-aware summation.
- Map each claimed computation to transformer primitives only after the proof story is clear enough to explain layer by layer.
- If an exact hand-coded design looks too deep or proof-heavy for the available time, preserve the option to report `P_1 = -1` rather than submit an unsupported claim.
- For `Problem 1-2`, optimize for smallest trainable architecture that clears the target under the fixed protocol.
- Use fixed positional encoding first because it is allowed and free under the counting rule.
- Sweep a small grid of depths / widths / heads and record both parameter count and final accuracy.
- Prefer architectures that are easy to explain and reproduce over brittle “just barely works” variants.
- Keep packaging constraints visible throughout the run:
- collect training curves and ablation notes while experimenting;
- draft the proof/report outline in parallel with implementation so the 2-page PDF does not become a last-minute blocker.

# Risks
- High: `Problem 1-1` exact multiplication may require more carry-management depth than expected, making a small hand-coded transformer hard to prove cleanly.
- High: an unsupported or hand-wavy proof for `Problem 1-1` receives no credit even if the weights appear numerically correct.
- High: bit-order mistakes are easy because both input and output are fixed-position LSB-first.
- Medium: `>=99%` may refer to exact-match sequence accuracy rather than token accuracy; undersized `Problem 1-2` models may fail if this is the judge’s metric.
- Medium: parameter counting disputes can invalidate “smallest” claims if tied weights, biases, or learned positional embeddings are counted incorrectly.
- Medium: the fixed training protocol removes tuning flexibility; some otherwise strong architectures may not converge within 200 epochs.
- Low: final packaging can fail if experiments are not logged and the team cannot reconstruct training curves or ablations for the report.

# Deliverables
- `runs/day1/codex/parser.md`
- This execution-ready brief for downstream Codex work.
- Execution Checklist
- Confirm the exact LSB-first tokenization with at least 3 worked examples including `63 × 63 = 3969`.
- Create a shared verifier for encode/decode correctness, greedy decoding, and unique-parameter counting.
- Produce a proof-oriented design note for `Problem 1-1`, including explicit layer/head/MLP responsibilities.
- Produce a trainable `build_model()` candidate list for `Problem 1-2` with parameter counts.
- Run and log a compact architecture sweep for `Problem 1-2` under the fixed training protocol.
- Record accuracy, model size, and failed attempts in a form usable for the PDF report.
- Prepare the final three-number submission fields: `P_1`, `P_2`, `Acc_2`.
- Prepare the final ZIP inputs: one PDF report and one single Python file.
- Builder Checks
- Do not start architecture work until the encode/decode contract and parameter-counting convention are frozen.
- Keep `Problem 1-1` proofability as a hard gate; a smaller but unexplainable design is not acceptable.
- Keep `Problem 1-2` experiments reproducible under the exact fixed training protocol rather than a tuned local variant.
- Verifier Checks
- Re-run token-order tests on every candidate model with greedy decoding, not teacher-forced reporting only.
- Validate that reported parameter counts use unique parameters after tying and include biases / learned positional parameters when present.
- Reject any candidate whose correctness argument or packaging artifacts cannot be traced back to explicit problem requirements.

# Immediate Next Actions
1. Build the shared encoding and verification harness first so every later result uses the exact required token format and counting rules.
2. Draft the `Problem 1-1` computation story as a proof outline before hand-coding any weights; reject designs that are hard to explain layer by layer.
3. Implement the smallest plausible `Problem 1-2` baseline with free positional encoding and run an initial training pass under the fixed protocol.
4. Sweep a few nearby `Problem 1-2` sizes, logging parameter count and final accuracy to identify the smallest model above the target.
5. Maintain a running report skeleton with proof notes, training curves, ablations, and packaging artifacts so final assembly is low-risk.
