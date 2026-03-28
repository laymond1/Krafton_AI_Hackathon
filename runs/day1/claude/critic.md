# Objective

Critical review v2 of updated `strategist.md` (v2, AdderBoard-informed), cross-checked against updated `codex/submission_draft.py`, `codex/builder.md`, `codex/adderboard_idea_map.md`, `merged/experiment_log.md`, `merged/solution_brief.md`, and `codex/verifier.md`.

---

# Facts from Problem

- Two sub-problems, both mandatory. P_1 = -1 is allowed for 1-1 but loses score.
- 1-1 proof is not optional — "weights without a proof receive no credit."
- 1-2: only `build_model()` submitted. Fixed training protocol: 200 epochs, AdamW lr=1e-3, cosine LR, batch=256, 100k samples. Target ≥99% exact match on 10k pairs.
- Parameter counting: fixed PE (sinusoidal, fixed-θ RoPE) does NOT count. Learned PE and all `nn.Parameter` count.
- Current state: no 1-1 implementation exists. All 1-2 candidates stuck at ~2.5% exact-match (3-10 epoch runs only). No 200-epoch run done.

---

# Assumptions Under Challenge

### A8 (RoPE period-6 aliasing) — PARTIALLY VALID, CONSEQUENCES MISJUDGED

The math is correct: with ω = 2π/6, positions 6 apart have identical rotational phase, so A_h (pos h) and B_h (pos h+6) alias. This is sound.

**But the consequence is wrong.** The strategist concludes: "Head h at output P_k attends most to A_h and B_h *regardless of k*." This would mean the attention output is IDENTICAL at every output position (positions 12+0 through 12+11). If the residual is the same at every output position, the MLP (with shared weights) produces the same output at every position. **The model would output the same bit 12 times.** This is clearly broken.

### A9 (200 epochs sufficient) — UNTESTED, HIGH RISK

No 200-epoch run exists. All experiments are 1-10 epochs. The gap between 2.5% (current) and 99% (target) is enormous. Whether longer training closes this gap depends entirely on model capacity, and all tested models are 28-142 params — likely far too small.

---

# Proposed Approach (Corrective Path)

## BLOCKER 1: 1-1 Option A has a fatal position-dependence gap

### The problem

The strategist's 1-1 Option A (1L, d=12, 6h MQA, RoPE period-6) relies on:
1. Attention "broadcasts" all (A_h, B_h) pairs into the residual at every output position.
2. MLP computes the correct column sum, selecting which pairs contribute via "PE(12+k) modulates which hidden units fire."

**These two claims contradict each other.** If attention routes the same (A_h, B_h) pairs at every output position (claim 1), then the residual is position-independent and the MLP has no position signal (invalidating claim 2). If attention varies with position (which RoPE causes), then claim 1 is wrong and the "broadcast all bits" strategy fails.

### Root cause: RoPE applies INSIDE attention, not to the residual

RoPE rotates Q and K vectors to create position-dependent attention patterns. But it does NOT inject position information into the residual stream. After attention, the residual at position p contains: `token_embed(p) + attention_output(p)`. The attention_output is position-dependent (because RoPE changed the attention weights), but in the strategist's design, if attention is "sharp" (each head attends to exactly one pair), the position dependency is negligible.

The MLP at position 12+k sees the residual and must compute COLUMN k's sum. With shared weights and no position signal, it cannot distinguish column 0 from column 5.

### Why sinusoidal PE solves this and RoPE doesn't

**Sinusoidal PE** is ADDED to the residual: `residual = token_embed + PE(pos)`. The MLP at position 12+k sees `token_embed(P_{k-1}) + PE(12+k) + attention_output(k)`. The PE(12+k) is different for each k, providing an explicit position signal. The MLP can use this to gate different hidden units for different columns.

**Sinusoidal PE is also FREE** (doesn't count toward parameters).

**Fix:** Replace RoPE with sinusoidal PE for 1-1. The MLP can then implement position-dependent gating: hidden unit for pair (i,j) fires when PE(12+k) matches i+j=k AND A_i=1 AND B_j=1.

### What happens to the RoPE aliasing trick?

Without RoPE, we lose the elegant (A_h, B_h) pair routing. We need a different attention strategy. Two options:

**Option 1 — "Broadcast all" with sinusoidal PE:**
- d_model ≥ 13 (12 bits + carry). Use sinusoidal PE (free).
- Attention at output positions copies all 12 input bits into the residual (need 12 heads or a clever V-projection).
- MLP uses the PE signal to compute the correct column sum.
- Carry propagates autoregressively.
- Estimated P_1: depends on attention mechanism, likely 500-2000.

**Option 2 — 2-layer with sinusoidal PE (strategist's Option B, now with PE fix):**
- d=4, 2 layers, sinusoidal PE (free).
- Layer 1 attention: copies relevant input bits via PE-matched attention.
- Layer 1 MLP: computes partial products using PE to identify pairs.
- Layer 2 attention: reads carry from previous output position.
- Layer 2 MLP: computes (S_k + C) mod 2, stores carry.
- Estimated P_1: 400-1000.

**Recommendation: Option 2 (2-layer, sinusoidal PE) is the most provable architecture for 1-1.** The layer separation gives modular proof structure. Sinusoidal PE provides position-dependent gating for the MLP.

---

## BLOCKER 2: No empirical path to 99% exact-match exists

### Current evidence

From `merged/experiment_log.md`:

| Candidate | Params | Epochs | Train size | Exact match | Bit accuracy |
|-----------|--------|--------|-----------|-------------|--------------|
| 3 (arc, d=3) | 51 | 10 | 100k | 0.0254 | 0.6273 |
| 5 (learned, d=4) | 142 | 3 | 100k | 0.0254 | 0.6219 |
| All others | 28-182 | 1-10 | 256-100k | 0.0-0.06 | 0.50-0.63 |

**Every model is stuck near random exact-match accuracy (1/4096 ≈ 0.024%).** The bit accuracy of 62% is only slightly better than always predicting 0 (which gives ~50% depending on bit distribution).

Codex per-bit analysis reveals: candidate 3 collapses to all-zeros; candidate 5 underpredicts 1s globally.

### Why these models fail

**Capacity is fundamentally insufficient.** Binary multiplication of 6×6 bits produces 12 output bits, each a complex nonlinear function of 12 input bits. The most complex output bit (P_5) depends on Σ(A_i · B_{5-i}) for 6 terms plus carry — this is a multi-input XOR with multi-bit carry, a function with high Boolean complexity.

A model with 28-142 parameters cannot represent this function faithfully. For comparison:
- The AdderBoard ADDITION problem: each output digit depends on 2 input digits + 1 carry bit. Carry is 0/1.
- Multiplication: each output bit depends on up to 6 partial products + carry up to 5. Much harder.

**The AdderBoard-to-multiplication scaling factor is not 1×.** The strategist assumes AdderBoard patterns scale directly, but the computational complexity is qualitatively different.

### The 200-epoch argument doesn't help if capacity is insufficient

200 epochs of training cannot make a 51-param model learn a function it cannot represent. The "grokking" phenomenon requires that the model has sufficient capacity to memorize the function first, then generalize. With 4096 distinct input-output pairs, the model needs enough capacity to express those patterns. 51 params is insufficient even for memorization.

### What capacity is needed?

**Lower bound estimate:** The output function has 12 bits × 4096 inputs = 49,152 bits of information. The model needs at least ~10k-20k parameters to have enough expressivity (rough rule: params × bits_per_param ≥ total_info × compression_factor).

**Upper bound from experience:** The original Codex scaffold (100k params) was also stuck at 2.5% in short runs. This doesn't tell us it can't converge in 200 epochs — it just hasn't been tested.

### Corrective path

1. **IMMEDIATELY run a 200-epoch fixed-protocol experiment** with the largest available model. Use Codex's candidate 5 (142 params) AND a d=32 standard model (17k params). This is the single most important action right now.
2. **If d=32 hits ≥99% at 200 epochs:** submit P_2 ≈ 17k, then search downward.
3. **If d=32 doesn't hit ≥99%:** escalate to d=64 (67k params) or d=128.
4. **Do not waste time on sub-200-param models** until a working baseline exists at any size.

---

## HIGH RISK 1: Claude strategist and Codex builder are architecturally divergent — no coordination

Claude strategist v2 proposes:
- 1-1: 1L d=12, 6h MQA, hd=2, RoPE period-6, ff=12 (P_1 ≈ 200-600)
- 1-2: Qwen3-style d=16, 2h/1kv GQA, SwiGLU, rank-3 factorization (P_2 ≈ 480-1328)

Codex builder is implementing:
- 1-1: not started
- 1-2: d=3-4 with various tying schemes, 28-142 params, all failing

**Neither system has a working solution. Their architectures are incompatible. There is no merge path.**

The `merged/comparison_table.md` is blank. The `merged/solution_brief.md` is stale (references the old 100k-param baseline).

**Corrective path:** The controller needs to make a single architectural decision for 1-2 NOW:
- Option A (conservative): d=32, 2L, 2h, ff=64, sinusoidal PE, standard GPT. P_2 ≈ 17k. High confidence of convergence. Run 200 epochs to confirm.
- Option B (aggressive): Try smaller models only AFTER Option A is confirmed working.

Both Claude and Codex should implement the SAME chosen architecture in their respective code, then compare results.

---

## HIGH RISK 2: The strategist's P_1 and P_2 estimates are overclaiming

### P_1 ≈ 200-600 is unsupported

The strategist's Option A claims P_1 ≈ 200-600 via "aggressive tying like AdderBoard." But no concrete weight-by-weight counting is provided. Let me estimate for the actual architecture (1L, d=12, 6h MQA, hd=2, ff=12):

- Token embedding: 2 × 12 = 24 (binary vocab)
- MQA with 6 query heads, 1 KV head:
  - Q: 6 × (12 × 2) = 144 (6 heads × d_model × head_dim)
  - K: 12 × 2 = 24 (shared KV head)
  - V: 12 × 2 = 24
  - O: 12 × 12 = 144
  - Total attention: 336
- MLP: up(12→12) + gate(12→12) + down(12→12) = 3 × 144 = 432 (SwiGLU)
- RMSNorm × 3: 3 × 12 = 36
- Output: tied to embed → 0
- **Raw total: 24 + 336 + 432 + 36 = 828 params (no tying beyond embed)**

With K=rot(Q), V=Q, O=Q^T: saves 24+24+144 = 192. Shared norms: saves 24. **With max tying: ~612 params.**

The 200-600 range is only achievable at the lower bound with extremely aggressive factorization that hasn't been verified. **More realistic: 600-850.**

### P_2 ≈ 480-1328 is unsupported AND likely won't converge

The strategist claims Qwen3-style with rank-3 factorization can reach P_2 ≈ 480. But:
1. No evidence that a 480-param model can learn 6-bit multiplication in 200 epochs.
2. All empirical evidence (Codex experiment log) shows even 142-param models fail completely.
3. The rank-3 factorization reduces effective capacity even further.

**Do not submit P_2 < 1000 without a confirmed ≥99% training run.**

---

## MEDIUM RISK 1: MLP mod-2 circuit complexity (carried over from v1, still valid)

For 1-1, computing P_k = T_k mod 2 where T_k = S_k + C_{k-1} ∈ {0,...,11}:
- The parity function on integers 0-11 needs at least 6 ReLU threshold units.
- The strategist's ff=12 is sufficient but the proof must explicitly construct the circuit.
- This is tractable but non-trivial under time pressure.

---

## MEDIUM RISK 2: Codex per-bit analysis reveals fundamental failure mode

The all-zero collapse (candidate 3) and global 1-underprediction (candidate 5) indicate the loss landscape for tiny binary-output models is adversarial:
- With vocab=2 and output ≈ 50% zeros, predicting all-zeros gives ~50% bit accuracy and cross-entropy ≈ log(2) ≈ 0.693.
- The models are stuck at local minima near this trivial solution.
- Breaking out requires either (a) more capacity, (b) better initialization, or (c) curriculum learning (not allowed under fixed protocol).

**This directly supports "go bigger first, optimize later."**

---

## LOW RISK: Codex `unique_parameter_count()` may not handle all tying patterns

The function checks `data_ptr()`:
```python
pointer = parameter.data_ptr()
if pointer in seen: continue
```

This works for shared `nn.Module` instances (same object = same pointer). But for derived parameters (e.g., `K = rotate(Q)` — K is computed from Q, not a stored parameter), the parameter count correctly counts only Q. However, if the rotate operation creates a new tensor, it's not registered as a parameter at all. **Verify:** in `k_rot_q` mode, is there any `nn.Parameter` for K? No — K is computed on the fly. So only Q's params are counted. This is correct.

---

# Risks (ordered by submission impact)

| Priority | Risk | Impact | Blocker? |
|----------|------|--------|----------|
| 1 | 1-1 Option A's position-gating is fundamentally broken (RoPE doesn't inject position into residual for MLP) | 1-1 architecture unimplementable as specified | YES |
| 2 | No model has come close to 99% exact-match. No 200-epoch run exists. | P_2 and Acc_2 are completely unsupported | YES |
| 3 | Claude and Codex architectures are divergent with no merge path | Duplicated effort, no single submit-ready artifact | HIGH |
| 4 | P_1 ≈ 200-600 and P_2 ≈ 480-1328 are overclaimed; evidence suggests both are larger | Report will contradict reality | HIGH |
| 5 | All tiny models (28-142 params) collapse to trivial solutions | Time wasted on models that can't work | HIGH |
| 6 | Mod-2 circuit needs explicit construction in proof | Proof incomplete without it | MEDIUM |
| 7 | merged/comparison_table.md blank, solution_brief.md stale | Controller decisions not traceable | MEDIUM |
| 8 | AdderBoard → multiplication scaling assumptions are unverified | Architecture decisions based on wrong analogy | MEDIUM |

---

# Deliverables

## Fix List (ordered by urgency)

### Immediate (do within next 15 minutes)

1. **Run a 200-epoch fixed-protocol experiment NOW.** Use d=32, 2L, 2h, ff=64, sinusoidal PE (the strategist's fallback Option B). This is the single most important action. If this hits ≥99%, we have a working P_2 baseline.

2. **In parallel, run 200-epoch on d=16** (Codex style with tying). If d=32 works but d=16 doesn't, we know the capacity floor.

### Within 30 minutes

3. **Fix 1-1 architecture: switch to sinusoidal PE, 2-layer design.** Abandon 1L RoPE period-6 Option A. Use 2-layer, d=4-8, sinusoidal PE, explicit carry propagation. The sinusoidal PE gives the MLP position-dependent gating for free (0 counted params).

4. **Align Claude and Codex on a single 1-2 architecture.** Controller picks one direction, both systems implement it. Stop parallel divergent experimentation.

### Within 60 minutes

5. **Start 1-1 weight coding** only after 1-2 is confirmed working. Code the 2-layer architecture, set weights, verify forward pass on worked examples.

6. **Update merged/comparison_table.md and solution_brief.md** with actual evidence-backed numbers.

### Within 90 minutes (hard cutoff)

7. **If 1-1 forward pass is not verified:** set P_1 = -1. Redirect all effort to proof quality for 1-2 and report writing.

8. **If no model hits ≥99% at 200 epochs:** escalate to d=64 or d=128 immediately. Do not spend more time on sub-1000-param models.

## Key insight for controller

**The AdderBoard analogy is misleading for model sizing.** Addition needs ~6-8 params. Multiplication is quadratically harder (up to 6 cross-product terms per output bit, multi-level carry). Expect P_2 to be 100-1000× larger than AdderBoard solutions. A working 17k-param model submitted on time beats a theoretically elegant 480-param model that doesn't converge.

## Output file

`runs/day1/claude/critic.md` — this document (v2).
