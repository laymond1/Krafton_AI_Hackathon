# Objective
- Verify the current Day 1 Codex path against the raw problem, focusing on `runs/day1/codex/submission_draft.py`, merged briefs, and final packaging readiness.
- Deliver a strict verification report plus a short fix queue that the controller can act on immediately.

# Facts from Problem
- Both `Problem 1-1` and `Problem 1-2` are required.
- `Problem 1-1` needs hand-coded transformer weights for exact multiplication plus a written layer-by-layer correctness proof.
- `Problem 1-2` needs a `build_model()` architecture that can be trained under the fixed protocol to `>=99%` accuracy.
- The submission must report three numbers: `P_1`, `P_2`, `Acc_2`.
- The final package must contain one max-2-page PDF report and one single Python file.
- Input and output formats are fixed 12-token LSB-first binaries, with greedy autoregressive decoding for output bits.

# Assumptions
- Accuracy is being treated as exact-match product accuracy rather than per-bit accuracy.
- Why it matters: current experiments are far below any safe submit claim under that interpretation.
- A truthful fallback of `P_1 = -1` is allowed if no exact `Problem 1-1` solution is found.
- Why it matters: the current repository does not contain a valid hand-coded exact solution or proof.
- The current smoke tests and short proxy runs are not equivalent to the fixed official `Problem 1-2` training protocol.
- Why it matters: no final `Acc_2` claim is currently supported.

# Proposed Approach
- Treat the current Codex state as a partially verified scaffold, not a submission-ready solution.
- Freeze all claims to what is currently evidenced:
- token format and greedy decode helpers are implemented and smoke-tested;
- short training proxies run successfully;
- no valid `Problem 1-1` artifact exists yet;
- no fixed-protocol `Problem 1-2` result exists yet.
- Use the next pass to close blockers in this order:
- make an explicit `Problem 1-1` decision (`solve now` or truthful fallback `P_1 = -1`);
- choose one concrete default `build_model()` configuration and keep docs aligned with it;
- run a more submission-like `Problem 1-2` experiment before claiming `P_2` or `Acc_2`;
- finish final packaging artifacts only after those decisions are locked.

# Risks
- Blocker: `Problem 1-1` is not solved in the current Codex path. There is no hand-coded exact-weight model and no layer-by-layer proof, so a normal `P_1` claim is unsupported.
- Blocker: `runs/day1/final/final_submission.md` is still a template, and there is no PDF report artifact. Packaging is not submission-ready.
- Blocker: no supported `Acc_2` exists. The logged runs are smoke tests and short proxy experiments, not the fixed 200-epoch protocol from the problem.
- Important: `build_model()` currently returns the large default model in `runs/day1/codex/submission_draft.py`, while the working notes discuss smaller candidate configs as the active search path. This is a sync risk for `P_2`.
- Important: the exact-match accuracy interpretation is still an assumption, not a stated fact. Final wording must keep that separation.
- Important: `runs/day1/merged/comparison_table.md` is blank, so controller-side merge decisions are not yet traceable.
- Optional: the standardized `wslim` Conda runtime imports PyTorch successfully, but it still reports `torch.cuda.is_available() = False`, so experiments remain CPU-bound.

# Deliverables
- Verification Report
- Verified today:
- `runs/day1/codex/submission_draft.py` should now be smoke-tested and run from Conda `wslim`, preferably through `./scripts/with_wslim.sh ...`.
- LSB-first encoding examples for `0 x 0`, `23 x 37`, and `63 x 63` are consistent with the problem statement.
- The current code path supports greedy decoding, exact-match evaluation, bitwise evaluation, and unique-parameter counting.
- Not yet verified:
- any exact `Problem 1-1` implementation;
- any `Problem 1-2` result under the official fixed protocol;
- final submission fields `P_1`, `P_2`, `Acc_2`;
- final PDF and zip-ready package contents.
- Fix Queue
- 1. Decide whether to continue `Problem 1-1` or lock truthful fallback `P_1 = -1`.
- 2. Pick the intended `build_model()` default and align code, experiment log, and solution brief.
- 3. Run a stronger `Problem 1-2` experiment before claiming `Acc_2`.
- 4. Fill `runs/day1/final/final_submission.md` with only supported claims.
- 5. Update comparison and packaging artifacts after the numerical claims are fixed.

# Immediate Next Actions
1. Create one controller-visible decision note for `Problem 1-1`: continue proof work or freeze `P_1 = -1`.
2. Mark `Problem 1-2` as `not yet submission-validated` in the final draft until a fixed-protocol-like run exists.
3. Sync the chosen `build_model()` default with the documented selected direction.
4. Update the final submission draft and comparison table only with evidence-backed numbers.
