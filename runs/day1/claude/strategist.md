# Objective

- Design the winning strategy for both Problem 1-1 (hand-coded, provable) and Problem 1-2 (trainable, ≥99% acc).
- Recommend concrete transformer architecture specs drawing from AdderBoard top-5 proven techniques.
- Minimize total parameter count while staying submit-ready under a 4-hour window.

---

# AdderBoard Deep Analysis (from actual prize code)

## How the 6-param / 8-param adder ACTUALLY works

Read from `first_place_prize.py` and `second_place_prize.py`:

### Architecture: 1L Qwen-style, d=2, 1h, hd=2, ff=2

**Embedding** (1-2 params):
- `e(d) = [c − d²/c, −d]` where c=1000
- dim0 ≈ constant (~1000 for all digits), dim1 = −digit_value
- For binary vocab: `e(0) = [c, 0]`, `e(1) = [c−1/c, −1]` → even simpler

**RoPE period-19 routing** (0 params, architectural constant):
- Input format: `[0] + A_reversed + [0]*9 + B_reversed + [0]` (31 tokens)
- A_k at position k+1, B_k at position k+20 → **19 apart**
- RoPE with ω = 2π/19 makes positions that are 19 apart have **identical rotational phase**
- Score(p,q) = cos(PHI − (p−q)·ω). Peaks at offset PHI/ω ≈ 10.3
- Offset to B_k from output: 11. Offset to A_k: 30 ≡ 11 (mod 19). **Both same score!**
- One head attends to BOTH A_k and B_k simultaneously

**V projection** (1 param): extracts dim1 (= −digit_value) → attention output ≈ −(A_k + B_k)

**SiLU carry hinge** (2+1 params):
- Gate: `g = a·dim0 + c·dim1` where dim1 ∝ −(digit_sum)
- SiLU(g) acts as a smooth threshold detector: fires when sum ≥ 10
- Carry signal propagates through autoregressive generation

**Output** (0 extra params): tied lm_head = embed^T, folded RMSNorm

### Core insight chain:
```
RoPE aliasing     →  route A_k, B_k to same head
V proj (scalar)   →  extract digit values
Attention sum      →  A_k + B_k  (the digit sum)
SiLU hinge         →  carry = 1 if sum ≥ 10
Autoregressive     →  carry propagates forward via context
```

## What changes for binary MULTIPLICATION

| Property | Decimal Addition (AdderBoard) | Binary Multiplication (our problem) |
|---|---|---|
| Vocab | 10 | **2** (trivial embedding) |
| Pairs per output bit | **1** (A_k, B_k) | **Up to 6** (all (A_i,B_j) where i+j=k) |
| Operation per pair | A_k + B_k (linear, attention does this!) | A_i · B_j (nonlinear, need MLP!) |
| Max carry | 1 (binary) | **5** (need multi-level detection) |
| Routing pattern | Fixed offset (A_k ↔ B_k always same distance) | **Variable** (different pairs at different offsets for each k) |

### Critical difficulty breakdown:

1. **Multi-pair accumulation**: Addition uses 1 pair per output; multiplication uses up to 6. Single-head RoPE routing targets ONE position pair. We need multi-head or multi-layer.

2. **Product vs sum**: Attention naturally computes weighted SUMS of V. For addition, sum IS the operation. For multiplication, we need AND products: `A_i · B_j = ReLU(A_i + B_j − 1)`. This requires the MLP nonlinearity.

3. **Variable routing**: In addition, the offset from output_k to (A_k, B_k) is constant → one RoPE angle suffices for all k. In multiplication, the relevant pairs (A_i, B_{k-i}) have offsets that vary with BOTH k and i. No single RoPE angle works for all.

4. **Multi-level carry**: Addition carry ∈ {0,1} → single SiLU hinge. Multiplication carry ∈ {0,...,5} → need multiple hinges or wider MLP.

---

## Hand-Coded Top 5 → Multiplication Adaptation

| # | Trick (from AdderBoard) | How to adapt |
|---|---|---|
| 1 | **d=2** (all top-5) | d must be larger. Target **d=12** (one dim per input bit) or **d=6** (one dim per A/B pair) |
| 2 | **RoPE period-19** (aliasing A_k↔B_k) | **RoPE period-6**: input A_h at pos h, B_h at pos h+6 → 6 apart. Offset 12 ≡ 6 (mod 6), so A_h and B_h alias to same phase |
| 3 | **Parabolic embed** e(d)=[c−d²/c,−d] | For binary: `e(0)=[c,0], e(1)=[c−1/c,−1]` → can simplify to scalar embed (1 param) |
| 4 | **Scalar V proj** (extract digit value) | Same: V extracts bit value (dim1 = −bit). But need separate extraction for A and B subsets |
| 5 | **SiLU carry hinge** (detect sum≥10) | **Multi-hinge**: need 2-3 hinge thresholds for carry values 0-5. ff=4-8 instead of ff=2 |
| 6 | **MQA with 5 heads** (entry #7, 28 params) | **6 heads MQA**: each head routes to (A_h, B_h) pair via period-6 aliasing |
| 7 | **Tied V/O, Q=angle** | Same tying principles. Q_h = angle_h (6 params for 6 heads, or structured to fewer) |

## Trained Top 5 → Multiplication Adaptation

| # | Trick (from AdderBoard) | How to adapt |
|---|---|---|
| 1 | **Circular arc embedding** (3 params) | Binary embed needs at most 2 params. e(b) = [r·cos(b·φ), r·sin(b·φ)] |
| 2 | **K=rotation(Q), V=Q** | Apply same tying. Reduces attention params by 50-66% |
| 3 | **SwiGLU** | Use SwiGLU for carry computation. Proven better than ReLU for arithmetic |
| 4 | **RoPE θ** (≈3 for addition) | For period-6: θ = 6/(2π) ≈ 0.955. Or tune empirically |
| 5 | **Rank-3 factorization** | Apply to all projections. Key for keeping trained P_2 small |
| 6 | **Shared all RMSNorms** | Same. One RMSNorm vector shared across all positions |
| 7 | **down=rotation(up^T)** | Same MLP tying. Reduces MLP params by 50% |

---

# Facts from Problem

- Two sub-problems, both required. P_1, P_2, Acc_2 are the three submitted numbers.
- Input: 12-token LSB-first binary sequence (A₀…A₅ B₀…B₅). Output: 12-token product P₀…P₁₁, autoregressive.
- Full causal sequence: 24 tokens. Model generates positions 12–23 one at a time.
- Architecture rules: Python + PyTorch, ≥1 self-attention layer, autoregressive, forward() = token IDs → logits, no conditional multiplication logic in code.
- Parameter counting: sinusoidal/fixed-θ RoPE PE does **not** count; bias terms **do** count.
- Problem 1-1: proof required, layer-by-layer. No proof = no credit.
- Problem 1-2: only `build_model()` submitted. Fixed training protocol (200 epochs, AdamW lr=1e-3, cosine LR, batch=256, 100k samples). Target ≥99% on 10k test pairs.

---

# Assumptions

Inherited from parser.md, updated:

- **A1**: Autoregressive at inference. Causal mask required.
- **A2**: Carry is implicit, must be encoded in residual stream.
- **A3**: Accuracy = exact match (all 12 bits correct per pair).
- **A5**: 1-1 and 1-2 architectures can differ.
- **A8 (new)**: RoPE period-6 aliasing can unify (A_h, B_h) routing, analogous to period-19 in AdderBoard.
- **A9 (new)**: For 1-2, 200 epochs may not be enough for "grokking" in very small models. Architecture must converge in standard training regime, not rely on long grokking.

---

# Proposed Approach

## Problem 1-1: Hand-Coded Weights

### Core algorithm

For output bit P_k: `P_k = (S_k + C_{k-1}) mod 2`, `C_k = (S_k + C_{k-1}) >> 1`
where `S_k = Σ_{i+j=k, 0≤i,j≤5} A_i · B_j` (column partial-product sum).

### The routing challenge (why multiplication ≠ addition)

In addition, RoPE routes 1 pair per output bit with a FIXED offset.
In multiplication, the relevant pairs (A_i, B_{k-i}) have offsets that **vary with k**.
No single fixed-offset RoPE routing handles all columns.

**Solution: "Broadcast all, compute locally"**

Instead of selective routing, use attention to gather ALL 12 input bits into the residual at every output position. Then let the MLP + positional encoding determine which column to compute.

---

### Option A: 6-head MQA + period-6 RoPE (AdderBoard-inspired, recommended)

**Architecture: 1L Qwen-style, d=12, 6h MQA, hd=2, ff=12, RoPE period=6**

**Embedding** (≤4 params with structure):
- `e(0) = [c, 0, c, 0, ..., c, 0]` (6 pairs of [c, 0])
- `e(1) = [c−1/c, −1, c−1/c, −1, ..., c−1/c, −1]`
- Parabolic structure: 2 params (c, curvature), tiled across d=12

**RoPE period-6 routing** (0 params):
- ω = 2π/6. At output pos 12+k:
  - Offset to A_h = 12+k−h. Offset to B_h = 12+k−(h+6) = 6+k−h.
  - Difference = 6 ≡ 0 (mod 6). **A_h and B_h alias to same rotational phase!**
- Head h: Q angle PHI_h tuned so peak offset = 12−h (targets A_h at output P_0, shifts with k)
  - Actually: score = cos(PHI_h − Δ·ω). For Δ = 12+k−h: score = cos(PHI_h − (12+k−h)·ω)
  - With period 6: cos(PHI_h − (12+k−h)·ω) = cos(PHI_h − ((12+k−h) mod 6)·ω)
  - (12+k−h) mod 6 = (k−h) mod 6 (since 12 ≡ 0 mod 6)
  - **Peak when (k−h) mod 6 ≡ PHI_h/ω**

- **Critical issue**: With period 6, positions 0-5 and 6-11 are indistinguishable! Head h peaks at both position h AND position h+6 (which is exactly what we want: A_h and B_h pair).

- So head h at output P_k attends most to **A_h and B_h** (regardless of k).

**After attention**:
- Head h output ≈ weighted sum of V at positions h and h+6
- With V projection structured to separate A/B: dim0 gets A_h value, dim1 gets B_h value
- Residual dims [2h, 2h+1] ≈ [A_h, B_h]

**MLP** (ff=12):
- Input: d=12 residual containing [A_0, B_0, A_1, B_1, ..., A_5, B_5] + PE(12+k) signal
- Hidden units compute partial products: `ReLU(A_i + B_j − 1)` = A_i AND B_j
- PE(12+k) modulates which hidden units contribute to which output bit
- Carry from previous output position read via autoregressive context in attention

**Why it could work:**
- Exact same RoPE aliasing trick as AdderBoard (period-6 instead of period-19)
- All 6 (A_h, B_h) pairs routed simultaneously via 6 MQA heads
- MLP computes AND products (binary-specific: ReLU(a+b−1))
- AdderBoard entry #7 (28 params) proves MQA with d=2, 5 heads is viable

**Why it could fail:**
- The MLP must implement position-dependent filtering (different column for different k) using PE signal. This is harder to hand-code than addition's position-independent carry hinge.
- Head h routes (A_h, B_h) pair, but column k needs pairs (A_i, B_{k-i}), NOT (A_h, B_h). **This means the attention routes the WRONG pairs!**

**FIX**: The attention broadcasts all (A_h, B_h) pairs. The MLP then selects the right combinations. For column k: the product A_i · B_j where i+j=k means we need A_i (from head i's dim0) times B_{k-i} (from head (k-i)'s dim1). The MLP hidden layer can compute: `ReLU(residual[2i] + residual[2(k−i)+1] − 1)` for each valid pair.

With d_ff=12 and PE-based gating, 6 hidden units handle the max column width. The PE at position 12+k encodes k, allowing the MLP to select which pairs to activate.

**Estimated P_1: 200–600 params** (with aggressive tying like AdderBoard)

**Proof structure:**
1. Embedding: show e(0), e(1) values
2. RoPE period-6: prove aliasing of offset-6 positions
3. Attention: prove head h attends to (A_h, B_h) pair with sharp weights
4. MLP: prove ReLU computes AND; show PE-gated selection for each column k
5. Carry: prove autoregressive propagation via attention to previous output

---

### Option B: 2-layer, d=4, carry-propagation (cleaner proof, larger P_1)

**Architecture: 2L decoder, d=4, 1h per layer, hd=4, ff=8, sinusoidal PE**

**Layer 1**: Attention copies a "summary" of relevant input bits. MLP computes raw column sum S_k for the current column using PE to identify which pairs contribute.

**Layer 2**: Attention reads carry from previous output position (12+k−1). MLP computes P_k = (S_k + C) mod 2, stores C_k for next position.

**Why it could work:**
- 2-layer separation gives cleaner proof structure (AdderBoard uses 1 layer, but we have harder problem)
- d=4 is small but sufficient for carrying partial sums + carry bits
- Proof: Layer 1 = "compute partial products", Layer 2 = "carry-aware output"

**Why it could fail:**
- With d=4, the residual must encode both the column sum (0-6) and position info — tight
- 2 layers doubles the parameter count

**Estimated P_1: 400–1000 params**

---

### Option C: 1-layer, d=14 "broadcast + compute" (safest proof)

**Architecture: 1L, d=14, 12h MQA, hd=2, ff=14**

12 heads, each routing to one specific input position (positions 0-11).
After attention: residual dims 0-11 contain all 12 input bit values. Dims 12-13 hold PE info and carry.
MLP: 14 hidden units compute up to 6 AND products per column, PE-gated.

**Why:** Most explicit proof — each head's role is trivially clear (attend to one position). MLP proof is pure boolean circuit.

**Why not:** d=14 with 12 heads is parameter-heavy. Estimated P_1: 800–2000.

---

### Recommendation for 1-1: **Option A first, Option B as fallback**

Option A directly extends the proven AdderBoard mechanism (MQA + RoPE aliasing) to multiplication. It has the lowest parameter count and follows the winning pattern. If the PE-gated MLP proof becomes too complex under time pressure, switch to Option B (cleaner 2-layer proof).

---

## Problem 1-2: Trainable Architecture

### Design philosophy (from AdderBoard trained top 5)

All top-5 trained entries use: **1-layer Qwen3-style, d=3, extreme weight tying, SwiGLU, RoPE**.

For multiplication: scale up appropriately but keep the same philosophy.

### Training protocol constraints

- Fixed: 200 epochs, AdamW(lr=1e-3, wd=0.01), cosine LR, batch=256, 100k samples
- Cannot use: curriculum learning, grokking-aware schedules, custom optimizers
- The model must converge to ≥99% within this standard regime
- This rules out very tiny models that require grokking (thousands of epochs)

---

### Option A: Qwen3-style, d=16, 1L, 2h/1kv, ff=32, SwiGLU + RoPE (recommended)

**Architecture (following AdderBoard winning pattern, scaled up):**
- Vocab=2, d=16, 1 layer, 2 query heads / 1 KV head (GQA), hd=8, ff=32
- RoPE with θ tuned for 24-token sequence (try θ ∈ {1, 3, 6})
- SwiGLU activation in MLP

**Weight tying (from AdderBoard trained top 5):**
- `K = rotation(Q)` → saves d×hd params
- `V = Q` → saves d×hd params
- `O = Q^T` → saves hd×d params
- `lm_head = embed^T` → saves d×2 params
- All RMSNorms shared → saves (n_norms−1)×d params
- `down_proj = rotation(up_proj^T)` → saves d×ff params

**Parameter count without tying:**
- embed: 2×16 = 32
- Q: 2×16×8 = 256; K: 16×8 = 128; V: 16×8 = 128; O: 2×8×16 = 256
- gate: 16×32 = 512; up: 16×32 = 512; down: 32×16 = 512
- 3× RMSNorm: 3×16 = 48
- lm_head: 16×2 = 32
- Total raw: ~2,416

**With tying (K=rot(Q), V=Q, O=Q^T, lm_head=embed^T, shared norm, down=rot(up^T)):**
- embed: 32 (tied to lm_head)
- Q: 256 (K, V, O all derived)
- gate: 512 (up derived via down=rot(up^T)... actually gate+up both needed for SwiGLU)
- For SwiGLU: gate_proj (16→32) + up_proj (16→32) + down_proj (32→16)
  - With down=rot(up^T): saves 512
  - gate: 512, up: 512, down: 0
- RMSNorm: 16 (shared)
- **Total: 32 + 256 + 512 + 512 + 16 ≈ 1,328 params**

**With rank-3 factorization on Q (from AdderBoard #3-5):**
- Q factored: 2×(16×3 + 3×8) = 2×72 = 144 instead of 256
- gate factored: 16×3 + 3×32 = 144 instead of 512
- up factored: same = 144
- **Total: 32 + 144 + 144 + 144 + 16 ≈ 480 params**

This would be extremely competitive if it converges in 200 epochs.

**Risk:** d=16 with rank-3 factorization may not have enough effective capacity for 99% exact match. But multiplication of 6-bit numbers has only 4096 distinct inputs, and 100k training samples gives ~24× coverage.

---

### Option B: Standard GPT, d=32, 2L, 2h, ff=64 (safe fallback)

**Architecture:** Standard pre-norm transformer, no exotic tying.
- d=32, 2 layers, 2 heads, ff=64, sinusoidal PE (free)
- Weight tying: embed↔lm_head only

**P_2 ≈ 17,000** (as estimated in v1)

**Why:** Guaranteed to converge. If Option A doesn't hit 99%, this is the safety net.

---

### Option C: Qwen3-style, d=8, 1L, 1h/1kv, hd=8, ff=16, SwiGLU (aggressive)

Same as Option A but smaller. With full AdderBoard-style tying:
- **P_2 ≈ 150–300 params**

**Risk:** Very likely needs grokking (>>200 epochs). Only try if time permits AND Option A is confirmed.

---

### Recommendation for 1-2: **Option A (d=16, ~480-1328 params), fallback to Option B (d=32, ~17k)**

Execution sequence:
1. Implement `build_model()` with Qwen3-style architecture, configurable d
2. Run d=16 with tying → check if ≥99% at epoch 200
3. If yes → try d=8 (Option C)
4. If no → run d=32 standard (Option B)

---

# Risks

1. **[CRITICAL] MLP position-gating for 1-1.** The MLP must compute different column sums for different output positions using the same weights. This relies on PE providing a clean signal. If the PE-gated MLP proof becomes unmanageable, fall back to Option B (2-layer).

2. **[HIGH] "Attention routes pairs, not products" gap.** Attention at head h gives (A_h + B_h), but we need A_i · B_{k-i}. The MLP must cross-reference dims from DIFFERENT heads. Verify this works before writing the full proof.

3. **[HIGH] Proof completeness under time pressure.** No proof = no credit. Budget at least 90 minutes for the proof. Start with representative cases (P_0, P_5, P_11).

4. **[HIGH] 1-2 convergence under fixed protocol.** With aggressive tying and small d, the model may not converge in 200 epochs. Run local verification FIRST before committing.

5. **[MEDIUM] Multi-level carry encoding.** AdderBoard carry is 0/1 (single SiLU hinge). Multiplication carry is 0-5, requiring either wider MLP or multi-threshold detection. Verify the ff width is sufficient.

6. **[MEDIUM] RoPE period-6 may cause unintended aliasing.** Positions 0,6,12,18 all alias. Output positions 12-23 alias with input positions 0-11. Need causal mask to prevent backward attention, and verify no spurious routing.

7. **[LOW] LSB-first encoding bugs.** Verify with worked examples: 23×37=851, 63×63=3969.

---

# Deliverables

| File | Owner | Status |
|------|-------|--------|
| `runs/day1/claude/strategist.md` | Strategist | ✅ this document (v2, AdderBoard-informed) |
| `runs/day1/claude/solution_1_1.py` | Builder | to build — spec: 1L d=12, 6h MQA, RoPE-6, ff=12 |
| `runs/day1/claude/solution_1_2.py` | Builder | to build — spec: Qwen3-style d=16, SwiGLU, aggressive tying |
| `runs/day1/claude/proof_1_1.md` | Writer | to write — layer-by-layer proof |
| `runs/day1/claude/writer.md` | Writer | to produce — 2-page report draft |

**Compact strategy for human controller approval:**

> **1-1**: 1L Qwen-style, d=12, 6-head MQA, hd=2, RoPE period-6 (aliasing A_h↔B_h), ff=12. Attention broadcasts all (A_h, B_h) pairs; MLP computes AND products via ReLU, PE-gated for column selection. Target **P_1 ≈ 200–600**. Fallback: 2L d=4, P_1 ≈ 400–1000.
>
> **1-2**: Qwen3-style 1L, d=16, 2h/1kv GQA, SwiGLU, rank-3 factorization, AdderBoard tying (K=rot(Q), V=Q, O=Q^T, shared norms, down=rot(up^T)). Target **P_2 ≈ 480–1328**. Fallback: standard d=32, P_2 ≈ 17k.

---

# Immediate Next Actions

1. **[Builder, NOW]** Implement 1-1 architecture skeleton: 1L, d=12, 6h MQA, hd=2, RoPE period-6, ff=12. Verify `forward()` runs on 24-token input. Use `second_place_prize.py` as code template (PyTorch, clean structure).
2. **[Builder, PARALLEL]** Implement 1-2 `build_model()`: Qwen3-style, d=16, SwiGLU, configurable tying. Run fixed training protocol locally on d=16 first.
3. **[Builder, NEXT]** Hand-code weights for 1-1:
   - Set RoPE ω = 2π/6
   - Set Q angles for 6 heads to route to positions h and h+6
   - Set V to separate A/B bit values into head subspaces
   - Set MLP to compute AND via ReLU + PE-gated column filtering
   - **Test on 23×37=851 and 63×63=3969**
4. **[Writer, AFTER step 3]** Write layer-by-layer proof for P_0, P_5, P_11.
5. **[Controller, 90min]** Compare with Codex output. If Codex has simpler 1-1 proof path, merge.
6. **[All, 180min]** Verify 1-1 on all 4096 pairs. Confirm 1-2 Acc_2 ≥ 0.99. Package.
