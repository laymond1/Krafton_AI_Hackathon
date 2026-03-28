# Objective
- Build the smallest transformer (by parameter count) that correctly multiplies two 6-bit binary numbers.
- Two sub-problems: (1-1) hand-coded weights with correctness proof, (1-2) trainable architecture achieving ≥99% accuracy.
- Submit three numbers (P_1, P_2, Acc_2) + a ZIP containing a PDF report (max 2 pages) and a single `.py` file.

---

# Facts from Problem

## Task
- Input: two integers a, b ∈ {0, 1, …, 63} (6-bit each).
- Output: product a × b, which fits in 12 bits (max 63×63 = 3969).
- Format: LSB-first binary. Vocabulary = {0, 1} (token IDs 0 and 1).
- Input sequence: A₀…A₅ B₀…B₅ (12 tokens, fixed positions, no separators).
- Output sequence: P₀…P₁₁ (12 tokens, LSB-first).
- Full sequence: 24 tokens. Model generates P₀…P₁₁ autoregressively given the 12-token input.
- Decoding: greedy (argmax) at every output position.

## Architecture constraints (all mandatory)
1. Python + PyTorch only.
2. At least one self-attention layer.
3. Autoregressive: model receives a token sequence, predicts next token one at a time.
4. `forward()` takes token ID tensor → returns logits. No if/else, lookup tables, or control flow that encodes multiplication logic.
5. Problem-specific knowledge lives only in weights, not code.

## What is allowed
- Any positional encoding: learned, sinusoidal, RoPE, ALiBi, etc.
- Any activation function: ReLU, GELU, SwiGLU, etc.
- Weight tying and parameter sharing.
- Low-rank / factorized projections.
- Custom embedding strategies for 2-token vocabulary.

## Parameter counting rules
- Count unique parameters after weight tying/deduplication.
- Fixed positional encodings (sinusoidal, RoPE with fixed θ) do NOT count.
- Learned positional encodings DO count.
- Bias terms count. All `nn.Parameter` values count.
- Rule of thumb: anything where `requires_grad=True` would apply in training counts.

## Problem 1-1 specifics
- Hand-code weights that perform exact multiplication.
- Must provide a written layer-by-layer correctness proof (attention head behavior + MLP computation + output composition).
- "A numerical dump of weights without explanation will receive no credit."

## Problem 1-2 specifics
- Submit only `build_model()` returning an untrained `nn.Module`.
- Fixed training protocol (cannot be modified):
  - Init: PyTorch defaults.
  - Data: 100,000 random pairs, a, b ∈ [0, 63].
  - Optimizer: AdamW(lr=1e-3, weight_decay=0.01).
  - Schedule: cosine annealing over 200 epochs, batch size 256.
  - Loss: cross-entropy on the 12 output-token positions only.
- Target: ≥99% accuracy on 10,000 randomly sampled test pairs.

## Submission format
| Field | Type | Description |
|-------|------|-------------|
| P_1   | int  | Parameter count, 1-1. -1 if no solution. |
| P_2   | int  | Parameter count, 1-2. |
| Acc_2 | float | Test accuracy 0.0–1.0, 1-2. |

ZIP contents:
1. PDF report (max 2 pages): architecture diagram, 1-1 proof, 1-2 training curve + accuracy vs. model size, ablations.
2. Single `.py` file defining the model and reproducing results.

---

# Assumptions

- **A1 (Autoregressive decoding boundary).** The 12 input tokens are provided as context; the model autoregressively generates positions 12–23. It is not stated whether the full 24-token sequence is fed as a single batch or whether teacher forcing is used during the output phase. *Assumption: during inference, we feed 12 input tokens then decode one token at a time, appending each predicted token to the context.* This matters for causal mask design.
- **A2 (Causal masking).** Standard autoregressive transformer requires a causal mask. The problem does not specify this explicitly but it is implied by "autoregressive." *Assumption: causal mask is required and the model cannot attend to future output positions.*
- **A3 (Accuracy metric is full-sequence exact match).** The problem says "10,000 pairs" and "accuracy" but does not define whether accuracy is per-bit or per-product. *Assumption: accuracy = fraction of pairs where all 12 output bits are correct (exact match), because the problem asks to "compute a × b."* If it is per-bit accuracy, ≥99% is achievable with far smaller models.
- **A4 (1-1 must use the same forward() interface).** The hand-coded model must also satisfy architecture rules 1–5. *Assumption: yes, same constraints apply.*
- **A5 (No shared architecture between 1-1 and 1-2).** The problem does not require the same architecture. *Assumption: they can be different, and the optimal strategy for each likely differs.*
- **A6 (Evaluation harness for 1-2).** The graders will call `build_model()`, then train with the fixed protocol, then evaluate. *Assumption: no hidden test set beyond random sampling of [0,63]×[0,63]; full coverage is 4096 pairs so 10k sampling with replacement is fine.*
- **A7 (Sinusoidal PE with fixed θ = 10000 is free).** Standard sinusoidal PE as in "Attention Is All You Need" does not count. *Assumption: this is the intended interpretation of "fixed positional encodings … do not count."*

---

# Proposed Approach

## High-level strategy
Two separate models. Minimize P_1 for 1-1 (provably correct, hand-coded). Minimize P_2 for 1-2 while hitting ≥99% accuracy under the fixed training protocol.

## Problem 1-1: Hand-coded weights
- **Algorithm to implement:** binary long multiplication via partial products.
  - P_k = XOR of all carries + partial products contributing to bit k.
  - More precisely: P = Σ_{i=0}^{5} Σ_{j=0}^{5} a_i · b_j · 2^{i+j}, where · is AND.
  - This is equivalent to: for each output bit k, P_k = (Σ_{i+j=k} a_i AND b_j + carry_in_k) mod 2.
- **Transformer role:** attention heads can compute pairwise AND products a_i · b_j; MLP layers handle XOR accumulation and carry propagation.
- **Key design question:** minimum number of layers and heads to implement carry-propagate addition of up to 6 partial products.
- **Approach:** design the smallest architecture where:
  1. Attention layer computes all 36 partial-product bits (a_i AND b_j) by attending over input positions.
  2. MLP layers implement multi-bit adder logic (XOR + carry).
- This likely requires 2–3 transformer layers minimum due to carry chain depth.

## Problem 1-2: Trained architecture
- Start with the smallest architecture that empirically reaches ≥99% in the fixed training budget (200 epochs, 100k samples).
- Hypothesis: a small transformer (1–2 layers, d_model ≈ 64–128, 2–4 heads) should be sufficient given the fixed and regular structure of multiplication.
- Use weight tying between embedding and output projection (token vocab = 2).
- Use sinusoidal PE (free parameters).
- Run ablations: vary d_model (32, 64, 128), layers (1, 2), heads (1, 2, 4).
- Select smallest model achieving ≥99% test accuracy.

---

# Risks

1. **[HIGH] Carry propagation depth for 1-1.** A transformer layer corresponds to one round of parallel computation. Binary addition with carry is O(log n) depth with carry-lookahead or O(n) depth naively. Getting exact carry propagation for 12-bit sums in a small number of layers is non-trivial. *Risk: hand-coded solution may require more layers than expected, increasing P_1.*
2. **[HIGH] Correctness proof quality for 1-1.** The problem explicitly says "weights without a proof receive no credit." The proof must be detailed and layer-by-layer. *Risk: if the architecture is complex, writing a complete proof under time pressure may be infeasible.*
3. **[MEDIUM] ≥99% accuracy under fixed training protocol.** The training protocol is fixed and not adjustable. 200 epochs with cosine annealing may be insufficient for some architectures. *Risk: chosen architecture for 1-2 may not converge to ≥99% within the fixed budget.*
4. **[MEDIUM] Exact match vs. per-bit accuracy interpretation.** If graders use per-bit accuracy, the bar is easier. If exact match, much harder for small models. *Risk: architecture too small fails exact-match target.*
5. **[LOW] Parameter counting disputes.** Edge cases around weight tying and bias terms. *Risk: graders count parameters differently.*
6. **[LOW] Token ID / LSB-first encoding bugs.** Easy to get byte-order wrong. *Risk: model produces correct bits in wrong order.*

---

# Deliverables

1. `runs/day1/claude/parser.md` — this document (problem brief).
2. `runs/day1/claude/strategist.md` — architecture strategy for 1-1 and 1-2.
3. `runs/day1/claude/solution_1_1.py` — hand-coded transformer with weights + correctness proof.
4. `runs/day1/claude/solution_1_2.py` — `build_model()` function for 1-2.
5. `runs/day1/claude/writer.md` — draft PDF report content.

---

# Immediate Next Actions

1. **Design the 1-1 architecture.** Determine the minimum transformer that can implement binary multiplication exactly. Start with the algorithm (partial products + addition) and map it to attention + MLP layers. Write a layer-by-layer execution trace for a small example (e.g., 3×5=15).
2. **Prototype 1-2 architecture.** Write `build_model()` with configurable d_model/layers/heads. Run the fixed training protocol on a small example to verify convergence behavior.
3. **Run ablation sweep for 1-2.** Test d_model = {32, 64, 128}, layers = {1, 2}, heads = {1, 2, 4} under the fixed protocol. Record accuracy and parameter count.
4. **Write 1-1 correctness proof.** Once architecture is chosen, write the layer-by-layer proof in parallel with code.
5. **Draft report.** Use results from steps 1–4 to fill the 2-page PDF.
