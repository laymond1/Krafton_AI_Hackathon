# Krafton AI Hackathon — 최종 제출용 문제 아이디어 10개

아래 아이디어들은 **MultiplierBoard (6-bit binary multiplication)** 기준으로,  
**Problem 1-2(학습형)** 중심으로 바로 실험 가능한 방향과,  
**Problem 1-1(핸드코딩 + 증명)** 에도 연결 가능한 방향을 함께 고려해 정리했다.

기본 전제:
- 입력: `A0..A5 B0..B5` (12 tokens, LSB-first)
- 출력: `P0..P11` (12 tokens, LSB-first, autoregressive)
- 핵심 구조: **partial product + column aggregation + carry propagation**
- 참고 기준: **AdderBoard leaderboard의 tiny transformer 압축 트릭**  
  (arc embedding, RoPE, tied output, shared norm, phase-tied routing, carry hinge 등)

---

## 1. Arc-Qwen Multiplier

### 핵심
AdderBoard trained 상위권의 구조를 가장 직접적으로 가져오는 방식이다.  
**1-layer tiny Qwen-style decoder** 위에
- circular arc embedding
- RoPE
- shared RMSNorm
- tied `O = Q^T`
- `K = rotation(Q)` 또는 `K = V`
를 사용해, 곱셈도 매우 작은 파라미터로 학습되게 노린다.

attention은 출력 위치별로 필요한 입력 비트 조합을 정렬하고,  
MLP는 partial product와 carry interaction을 압축하는 역할을 맡는다.

### 장점
- 가장 검증된 tiny-trained 계열에서 출발하므로 실패 확률이 낮다.
- fixed positional encoding을 쓰면 positional params를 거의 공짜로 가져갈 수 있다.
- sweep가 쉽고, 작은 변형(`K=V`, `K=rotation(Q)`, ff=2/3) 비교가 빠르다.
- 실험 중인 workspace에서 Codex가 바로 구조 템플릿으로 만들기 좋다.

### 리스크
- addition에서 검증된 구조라도 multiplication은 partial product 수가 많아 더 어렵다.
- 너무 작게 가면 per-bit는 맞아도 exact-match 99%를 못 넘길 수 있다.
- carry 폭발이 심한 dense×dense 입력에서 불안정할 수 있다.

---

## 2. Diagonal Partial-Product Transformer

### 핵심
곱셈의 본질을 가장 정직하게 반영한 구조다.  
출력 비트 `P_k`는 사실상 `i + j = k`를 만족하는 모든 `(A_i, B_j)` partial product와 carry의 함수이므로,  
attention이 **같은 diagonal(column)** 에 속한 비트들을 모으게 설계한다.

즉, addition의 “digit alignment”를 multiplication의 “diagonal alignment”로 바꾼다.

### 장점
- 문제 구조와 직접 대응되어 설명력이 높다.
- hand-coded proof와 trained architecture 모두 같은 스토리로 밀 수 있다.
- 보고서에 “왜 이 구조가 곱셈에 자연스러운가”를 쓰기 좋다.
- per-column error analysis와도 잘 맞는다.

### 리스크
- 한 column에 들어오는 partial product 개수가 위치마다 달라서 uniform한 tiny model이 다루기 어렵다.
- 중앙 column에서 carry complexity가 급증한다.
- phase alignment를 잘 못 주면 그냥 일반 decoder보다도 비효율적일 수 있다.

---

## 3. Carry-Save Multiplier Transformer

### 핵심
곱셈을 한 번에 최종 bit로 만들지 말고,  
중간에 **sum-like stream**과 **carry-like stream**으로 분해해 처리하는 2-stage 아이디어다.

예:
- Layer 1: partial products를 local aggregation 해서 중간 합 표현 생성
- Layer 2: 이 중간 표현을 바탕으로 autoregressive final carry propagation 수행

즉, 일반적인 schoolbook multiplication을 내부적으로 carry-save 스타일로 근사한다.

### 장점
- multiplication의 가장 어려운 부분인 “동시에 많은 partial product를 더하는 문제”를 완화한다.
- trained model이 학습하기 더 쉬워질 가능성이 높다.
- 2-layer를 써도 각 layer 역할이 뚜렷해 설명이 좋다.

### 리스크
- 1-layer보다 파라미터가 늘어난다.
- intermediate representation이 실제로 잘 분리되지 않으면 그냥 깊기만 한 모델이 된다.
- hand-coded exact proof로 이어가기는 상대적으로 까다롭다.

---

## 4. Proof-First Column Adder

### 핵심
Problem 1-1용으로 특히 적합한 아이디어다.  
작게 만드는 것보다 **레이어별 증명 가능성**을 최우선으로 둔다.

구성 예:
- Head A: 현재 출력 위치 `k`에 필요한 partial products 선택
- Head B: 이전 출력들에서 carry-relevant context 읽기
- MLP 1: local AND / count-like transform
- MLP 2: `sum + carry_in -> (P_k, carry_out)` thresholding
- Output head: 최종 0/1 decode

### 장점
- “각 attention head가 무엇을 보고, 각 MLP가 무엇을 계산하는가”를 명확히 서술할 수 있다.
- 보고서의 layer-by-layer proof 섹션 작성이 쉽다.
- hand-coded 제출에서 가장 중요한 설명 가능성을 확보한다.

### 리스크
- trained 쪽 최적 구조와는 다를 수 있다.
- 파라미터 수는 공격적으로 작아지지 않을 가능성이 높다.
- 실제 analytic weights를 만드는 난이도는 여전히 높다.

---

## 5. Two-Hinge ReLU Column Threshold Network

### 핵심
AdderBoard hand-coded 쪽에서 잘 먹힌 **carry hinge** 아이디어를 곱셈에 맞게 확장한다.  
곱셈의 각 column은 partial product의 개수와 carry에 따라 bit와 다음 carry가 결정되므로,
핵심 비선형은 결국 threshold 함수들이다.

예:
- `sum + carry_in >= 1`
- `sum + carry_in >= 2`
- `sum + carry_in >= 4`
같은 조건을 ReLU hinge 조합으로 근사 또는 구현한다.

### 장점
- Problem 1-1의 constructive proof에 매우 잘 맞는다.
- MLP의 역할을 threshold logic으로 명확히 설명할 수 있다.
- 작은 feedforward로도 surprisingly expressive할 수 있다.

### 리스크
- trained setting에서는 SGD가 이런 hinge 구조를 잘 못 찾을 수 있다.
- column마다 필요한 threshold 패턴이 달라져 tie가 어려울 수 있다.
- exact implementation은 수치 안정성이 민감할 수 있다.

---

## 6. Bit-Plane Factorization Transformer

### 핵심
곱셈을 “B의 각 비트가 켜질 때 `A << j`를 더하는 문제”로 본다.  
즉, `B_j`가 gate 역할을 하고 `A_i`는 shift/copy 대상이 된다.

attention/head specialization 예:
- one head: `B_j` on/off detection
- another head: `A_i` content routing
- final block: 여러 shifted planes를 합산

### 장점
- binary multiplication의 정의와 직접 맞닿아 있다.
- `A`와 `B`의 역할이 분리되어 해석이 쉽다.
- 특정 head가 특정 plane을 담당하는 식의 specialization을 기대할 수 있다.

### 리스크
- 여러 plane을 합치는 순간 carry 문제가 다시 어렵게 등장한다.
- 너무 구조를 나누면 tiny regime에서 오히려 비효율적일 수 있다.
- 1-layer보다는 2-layer 이상이 더 자연스러울 가능성이 있다.

---

## 7. Population-Count Column Network

### 핵심
각 output column을 “작은 popcount 문제”로 보는 관점이다.  
즉, `i+j=k`에 해당하는 partial product 비트들의 합과 carry_in이 있으면,
`P_k`는 parity, carry_out은 quotient 계열이다.

attention은 같은 column의 partial products를 모으고,  
MLP는 작은 popcount / parity / carry extractor 역할을 한다.

### 장점
- binary라서 count semantics가 매우 자연스럽다.
- multiplication의 구조를 직접적으로 포착한다.
- per-column error analysis와 연결하기 좋다.
- sparse activation(ReLU²류)와 잘 맞는다.

### 리스크
- soft attention이 실제 discrete count를 얼마나 잘 근사하는지가 관건이다.
- 중앙 column은 popcount 범위가 더 넓어 단일 MLP가 부족할 수 있다.
- exact-match 성능을 높이려면 carry handling이 추가로 필요하다.

---

## 8. Phase-Tied Diagonal Router

### 핵심
AdderBoard hand-coded 상위권의 **phase-tied Q projection** 아이디어를,  
곱셈의 diagonal routing에 맞게 재해석한다.

목표는 출력 위치 `k`가
- 관련 없는 비트는 무시하고
- `i+j=k`에 해당하는 입력 비트쌍 또는 그에 준하는 representation만 강하게 보게 하는 것

즉 positional phase를 이용해 “곱셈 column selector”를 만든다.

### 장점
- parameter tying이 잘 되면 매우 작게 만들 수 있다.
- positional structure를 공짜에 가깝게 활용할 수 있다.
- hand-coded와 trained 사이의 브리지 아이디어가 된다.

### 리스크
- 설계가 잘못되면 routing만 되고 실제 arithmetic이 안 된다.
- proof-friendly하긴 하지만 exact weights를 찾는 건 여전히 어렵다.
- trained에서는 과도한 tying이 표현력을 죽일 수 있다.

---

## 9. Sparse Output-Step Specialist Decoder

### 핵심
autoregressive decoding의 각 output step `k`가 사실상 필요한 정보만 보게 만드는 방향이다.  
즉, 모든 token을 다 equally 처리하는 대신,
현재 `P_k`를 만들 때 중요한 입력/이전출력만 강하게 보도록 학습 bias를 준다.

구현 힌트:
- RoPE/ALiBi/relative bias 기반 정렬 강화
- tied attention patterns
- shared minimal MLP

### 장점
- autoregressive 본질과 잘 맞는다.
- 작은 모델이 “불필요한 연산”을 줄이게 도와준다.
- exact-match 성능을 높이는 데 유리할 수 있다.

### 리스크
- 문제특화 mask를 코드에 넣으면 규칙 위반 소지가 있어 구조적으로 조심해야 한다.
- 너무 sparse하면 예외 케이스에서 정보가 부족할 수 있다.
- 학습이 불안정하면 routing조차 무너지기 쉽다.

---

## 10. Dual-Track Submission Strategy Model Family

### 핵심
하나의 모델로 모든 걸 해결하려 하지 않고,  
**Problem 1-1용 family**와 **Problem 1-2용 family**를 분리해서 설계한다.

권장 분리:
- Problem 1-2: **Arc-Qwen Multiplier / Diagonal Tiny**
- Problem 1-1: **Proof-First Column Adder / Two-Hinge Threshold**
- 공통: encode/decode, evaluation, parameter counting, report framing

즉 아이디어 자체를 모델 하나가 아니라 “제출 전략 패키지”로 본다.

### 장점
- 실전 해커톤에서 가장 합리적이다.
- trained 쪽 성과가 proof track에 발목 잡히지 않는다.
- hand-coded는 proofability, trained는 score efficiency에 맞게 최적화할 수 있다.
- final ZIP 정리도 수월하다.

### 리스크
- 코드 구조를 깔끔하게 유지하지 않으면 한 파일 제출이 복잡해질 수 있다.
- 두 트랙을 동시에 관리하므로 실험 로그 discipline이 필요하다.
- hand-coded 쪽이 끝까지 미완성일 수 있다.

---

# 추천 우선순위

실험 순서 추천:

1. **Arc-Qwen Multiplier**
2. **Diagonal Partial-Product Transformer**
3. **Carry-Save Multiplier Transformer**
4. **Proof-First Column Adder**
5. **Two-Hinge ReLU Column Threshold Network**

---

# Codex 입력용 짧은 요약

- addition의 digit alignment를 multiplication의 diagonal alignment로 바꿔라.
- tiny trained baseline은 1-layer Qwen-style, d=3, arc embedding, RoPE, shared RMSNorm, tied `O=Q^T`부터 시작하라.
- hand-coded는 proofability 우선으로 head별 역할과 MLP threshold logic이 설명 가능한 구조를 택하라.
- multiplication은 partial product aggregation과 carry propagation이 핵심 병목이므로, per-column / per-bit error analysis를 반드시 넣어라.
- 하나의 모델로 두 문제를 동시에 해결하려 하지 말고, trained family와 proof family를 분리해서 운영하라.
