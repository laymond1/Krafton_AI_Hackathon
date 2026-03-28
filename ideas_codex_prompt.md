# Codex 실행 프롬프트

이 문서는 `ideas_from_paper_and codes.md`와 `papers/` 아래 논문 자료를 바탕으로, 이 레포에서 바로 실행 가능한 **Codex용 작업 프롬프트**를 정리한 것이다.

목표:
- Krafton AI Hackathon MultiplierBoard `(6-bit x 6-bit binary multiplication)`에 직접 맞는 아이디어만 사용
- 논문 요약보다 **구현, 평가, 비교, 다음 결정**에 집중
- 현재 레포 구조와 `runs/day1/` 산출물 흐름에 맞춰 바로 작업 가능하게 만들기

참고 자료:
```text
./ideas_from_paper_and codes.md
./papers/idea1_2_5_arithmetic/arithmetic.txt
./papers/idea1_2_5_arithmetic/NeurIPS-2024-transformers-can-do-arithmetic-with-the-right-embeddings-Paper-Conference.pdf
./papers/idea3_teaching_arithmetic/teaching_arithmetic.txt
./papers/idea3_teaching_arithmetic/ICLR-2024-Teaching Arithmetic to Small Transformers.pdf
./papers/idea4_dissect_multi/dissect_multi.txt
./papers/idea4_dissect_multi/Dissecting Multiplication in Transformers.pdf
```

문제 전제:
- 입력: `A0..A5 B0..B5` 총 12개 bit, `LSB-first`
- 출력: `P0..P11` 총 12개 bit, `LSB-first`, autoregressive decode
- 최소 1개 이상의 self-attention layer 필요
- 핵심 병목: `partial product aggregation`, `diagonal routing (i+j=k)`, `carry propagation`

아래 프롬프트를 Codex에 그대로 넣어 작업시키면 된다.

```text
You are working inside the local repository for the Krafton AI Hackathon multiplier task.

Your goal is not to summarize papers. Your goal is to turn the paper ideas into concrete code, evaluation, and experiment-ready artifacts for this repository.

First read these local files for project context:
1. ./README.md
2. ./base.md
3. ./ideas_from_paper_and codes.md
4. ./runs/day1/codex/builder.md
5. ./runs/day1/codex/submission_draft.py

Then read and use these local references:
6. ./papers/idea1_2_5_arithmetic/arithmetic.txt
7. ./papers/idea1_2_5_arithmetic/NeurIPS-2024-transformers-can-do-arithmetic-with-the-right-embeddings-Paper-Conference.pdf
8. ./papers/idea3_teaching_arithmetic/teaching_arithmetic.txt
9. ./papers/idea3_teaching_arithmetic/ICLR-2024-Teaching Arithmetic to Small Transformers.pdf
10. ./papers/idea4_dissect_multi/dissect_multi.txt
11. ./papers/idea4_dissect_multi/Dissecting Multiplication in Transformers.pdf

Task context:
- Input prompt bits: A0..A5 B0..B5, zero-padded, LSB-first
- Output bits: P0..P11, zero-padded, LSB-first
- Sequence is autoregressive
- We want the smallest trainable architecture possible for Problem 1-2
- We also want analysis hooks that help proof-oriented design for Problem 1-1

Working rules:
- Prefer editing the existing scaffold in ./runs/day1/codex/submission_draft.py instead of creating a separate large framework.
- Keep helper analysis outside the forward pass.
- Do not add problem-specific Python control flow inside forward() that would violate the competition rules.
- Do not overwrite unrelated local changes.
- Use the repository's current day1 paths for outputs and notes.
- If a paper idea is not directly useful for 6-bit multiplication, ignore it.

Priority ideas to adapt from the papers:
1. Stronger position handling from arithmetic embedding work
   - make A-side and B-side positions explicitly distinguishable
   - preserve or introduce fixed/free positional structure when helpful
   - make diagonal structure i+j=k easier to learn

2. Shared recurrent block / input injection
   - create a tiny shared block variant reused 2 or 3 times
   - optionally re-inject original token representations each recurrence
   - compare unique parameter count against a non-shared tiny baseline

3. Intermediate structure for analysis, not final I/O
   - partial products
   - diagonal sums
   - carry traces
   Use these only for debugging, metrics, optional auxiliary experiments, or architecture comparison.

4. Specialization across attention heads or hidden subspaces
   - one path for diagonal evidence gathering
   - one path for carry/context retrieval from earlier outputs
   - MLP or subspace for popcount-like or threshold-like mixing

5. Evaluation harness first
   - exact-match accuracy
   - per-output-bit accuracy for P0..P11
   - error rate by output column
   - dense x dense input slice
   - carry-chain difficulty or carry-count slice if measurable

Concrete implementation tasks:
1. Upgrade ./runs/day1/codex/submission_draft.py so it includes:
   - reusable encode/decode helpers
   - greedy decode
   - unique parameter counting
   - richer evaluation with per-column metrics
   - easy comparison across candidate configs

2. Implement at least two architecture variants:
   - Variant A: position-strengthened tiny baseline
   - Variant B: shared/recurrent tiny variant with optional input injection

3. Add analysis helpers for:
   - product bit difficulty by column
   - dense/sparse operand slices
   - optional carry-oriented diagnostics

4. Keep the code submission-friendly:
   - modular but still compact
   - easy to collapse into a single-file final submission
   - comments should say which paper idea is being adapted and why

5. Update repo artifacts after code changes if appropriate:
   - ./runs/day1/merged/experiment_log.md
   - ./runs/day1/merged/solution_brief.md
   Only write what is actually supported by code or measurements.

6. Run lightweight verification when possible using the repository environment.
   Prefer the repo wrapper if needed:
   - ./scripts/with_wslim.sh python runs/day1/codex/submission_draft.py --smoke-test

Expected deliverables:
- improved ./runs/day1/codex/submission_draft.py
- at least 2 candidate architecture configs
- evaluation output that includes exact match and per-column behavior
- short markdown summary of:
  - what changed
  - which paper idea each change came from
  - parameter counts
  - exact-match accuracy
  - per-column failure patterns
  - recommended next move

How to report back:
- First list concrete code or artifact changes.
- Then list findings and risks.
- Keep paper summary short and only include points that directly changed the implementation.
- If you could not run full training or evaluation, say exactly what was verified and what remains unverified.
```
