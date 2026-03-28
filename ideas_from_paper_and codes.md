# MultiplierBoard 적용 아이디어 요약

이 문서는 아래 로컬 자료를 바탕으로, **Krafton AI Hackathon MultiplierBoard (6-bit binary multiplication)** 에 바로 적용 가능한 아이디어만 압축 정리한 것이다.

참고 디렉토리:
```text
./papers
├── idea1_2_5_arithmetic
│   ├── arithmetic
│   ├── arithmetic.txt
│   └── NeurIPS-2024-transformers-can-do-arithmetic-with-the-right-embeddings-Paper-Conference.pdf
├── idea3_teaching_arithmetic
│   ├── ICLR-2024-Teaching Arithmetic to Small Transformers.pdf
│   ├── teaching_arithmetic
│   └── teaching_arithmetic.txt
└── idea4_dissect_multi
    ├── dissect_multi.txt
    └── Dissecting Multiplication in Transformers.pdf
```

문제 전제:
- 입력: `A0..A5 B0..B5` (12 tokens, LSB-first)
- 출력: `P0..P11` (12 tokens, LSB-first, autoregressive)
- 핵심 병목: **partial product aggregation + carry propagation**
- 목표: tiny trained architecture를 빠르게 찾고, hand-coded/proof 방향에도 연결 가능한 설계 포인트를 확보

---

## 적용 아이디어 5개

## 1. Abacus-style 위치 구분 강화

### 요지
`Transformers Can Do Arithmetic with the Right Embeddings`에서 핵심은 arithmetic 실패의 상당 부분이 **숫자 내부 위치 추적 실패**에서 온다는 점이다.  
우리 문제는 고정 길이 6-bit × 6-bit 이진수라 더 단순하지만, 여전히 모델이 `A_i`와 `B_j`의 내부 위치를 안정적으로 구분해야 `i + j = k` diagonal 구조를 제대로 학습할 수 있다.

### 바로 적용 방식
- learned absolute positional embedding 대신 **고정 positional scheme**부터 사용
- operand-aware positional feature를 추가해 `A`쪽 6칸과 `B`쪽 6칸을 명확히 분리
- output `P_k`가 사실상 `i+j=k` 열을 찾도록 유도
- `A/B`를 같은 vocab token으로만 두지 말고, **position side 정보는 residual/feature 쪽에서 분리**

### 기대 효과
- tiny model이 자리 혼동을 덜 함
- diagonal routing이 쉬워짐
- learned positional parameter를 줄이거나 없애기 쉬움

### 연결 자료
- `./papers/idea1_2_5_arithmetic/arithmetic.txt`
- `./papers/idea1_2_5_arithmetic/NeurIPS-2024-transformers-can-do-arithmetic-with-the-right-embeddings-Paper-Conference.pdf`
- `./papers/idea1_2_5_arithmetic/arithmetic`

---

## 2. Shared recurrent block / input injection

### 요지
같은 논문 계열에서, 위치 구분이 안정되면 **recurrent layer**나 **input injection**이 arithmetic에 추가 이득을 주는 방향이 제시된다.  
우리 문제에서는 “큰 블록 1번”보다 “아주 작은 shared block을 여러 번 적용”하는 쪽이 parameter budget 대비 더 유리할 수 있다.

### 바로 적용 방식
- 1-layer giant block 대신 **1 tiny block + recurrence 2~3회**
- recurrence마다 역할을 느슨하게 분담:
  1. diagonal evidence 수집
  2. partial product aggregation
  3. carry-like state refinement
- 입력 원본을 각 recurrence에 약하게 재주입(input injection)해서 정보 소실을 줄임

### 기대 효과
- 파라미터 공유로 크기를 줄이면서 연산 깊이는 확보
- 곱셈처럼 carry가 복잡한 연산에 더 적합할 수 있음
- 2-layer 대비 더 작은 unique parameter count 가능성

### 연결 자료
- `./papers/idea1_2_5_arithmetic/arithmetic.txt`
- `./papers/idea1_2_5_arithmetic/NeurIPS-2024-transformers-can-do-arithmetic-with-the-right-embeddings-Paper-Conference.pdf`
- `./papers/idea1_2_5_arithmetic/arithmetic`

---

## 3. Intermediate representation을 학습 탐색용으로 활용

### 요지
`Teaching Arithmetic to Small Transformers`의 핵심은 **데이터 포맷**과 **intermediate-step supervision**이 성능과 sample efficiency를 크게 바꾼다는 점이다.  
최종 제출 포맷은 고정이지만, 로컬 탐색 단계에서는 어떤 architecture가 곱셈의 내부 구조와 잘 맞는지 판단하기 위해 intermediate representation을 활용할 수 있다.

### 바로 적용 방식
최종 제출 모델 형식은 그대로 두되, 탐색 단계에서만 아래 정보를 분석/보조학습에 사용:
- partial products
- diagonal sums
- carry trace
- 각 output bit가 참조해야 할 입력 위치 집합

보조 목적은 “최종 제출을 바꾸는 것”이 아니라:
- 어떤 구조가 곱셈 내부 구조를 잘 담는지 찾기
- failure mode를 더 빨리 식별하기
- architecture search를 덜 블라인드하게 만들기

### 기대 효과
- 단순 exact-match 수치보다 빠르게 방향성 판단 가능
- tiny 모델의 실패 원인을 구조적으로 파악 가능
- Codex가 실험 루프를 만들 때 디버깅 효율이 올라감

### 연결 자료
- `./papers/idea3_teaching_arithmetic/teaching_arithmetic.txt`
- `./papers/idea3_teaching_arithmetic/ICLR-2024-Teaching Arithmetic to Small Transformers.pdf`
- `./papers/idea3_teaching_arithmetic/teaching_arithmetic`

---

## 4. Head / subspace를 diagonal-carry subtask로 분업

### 요지
`Dissecting Multiplication in Transformers`는 transformer가 곱셈을 여러 **parallel subtasks**로 나눠 학습하고, 특히 병목이 **successive carryovers**와 **intermediate result caching**에 있다는 점을 보여준다.  
이건 우리 문제에서 attention head 또는 hidden subspace를 “그냥 전체 연산”에 쓰지 말고, 역할을 나눠 설계해야 한다는 힌트다.

### 바로 적용 방식
예시 분업:
- Head 1: `i+j=k`에 가까운 diagonal evidence gathering
- Head 2: 이전 출력 토큰들에서 carry-relevant context retrieval
- MLP / subspace A: local popcount-like mixing
- MLP / subspace B: threshold/carry refinement
- residual stream: intermediate cache 역할

### 기대 효과
- multiplication의 핵심 어려움을 구조적으로 반영
- hand-coded proof story와도 연결 가능
- per-head probing/분석이 쉬워짐

### 연결 자료
- `./papers/idea4_dissect_multi/dissect_multi.txt`
- `./papers/idea4_dissect_multi/Dissecting Multiplication in Transformers.pdf`

---

## 5. Per-column / carry-chain 평가 하네스를 먼저 붙이기

### 요지
가장 실용적인 아이디어다.  
작은 모델은 overall accuracy 하나만 보면 왜 실패하는지 알기 어렵다. 특히 multiplication은 중앙 column과 긴 carry chain에서 많이 무너질 가능성이 높다.  
그래서 architecture search 초기에 **정밀 error analysis 하네스**를 먼저 붙이는 게 실제 성능 향상에 가장 직접적이다.

### 바로 적용 방식
exact-match accuracy 외에 반드시 함께 기록:
- `P_k`별 bit accuracy
- dense×dense 입력에서의 성능
- 중앙 column(`k≈5~7`) 오류율
- carry-chain 길이별 오류
- `A` sparse / `B` dense 같은 분포별 오류

### 기대 효과
- 어떤 아이디어가 실제로 병목을 줄였는지 빨리 판별 가능
- tiny architecture sweep가 훨씬 효율적이 됨
- hand-coded/proof 방향에서도 어떤 column logic이 어려운지 파악 가능

### 연결 자료
- 주된 아이디어 출처:
  - `./papers/idea1_2_5_arithmetic/arithmetic.txt`
  - `./papers/idea1_2_5_arithmetic/arithmetic`
- 보조 해석:
  - `./papers/idea4_dissect_multi/dissect_multi.txt`

---

# 실전 우선순위

추천 적용 순서:

1. **Per-column / carry-chain 평가 하네스 추가**
2. **Abacus-style 위치 구분 강화**
3. **Head / subspace를 diagonal-carry subtask로 분업**
4. **Shared recurrent block / input injection 실험**
5. **Intermediate representation 기반 로컬 분석/보조학습**

---
