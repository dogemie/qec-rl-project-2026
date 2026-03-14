# 🧪 Simulation Module (`sim/`)

이 디렉토리는 AlphaZero 에이전트가 생성한 양자 오류 정정(QEC) 패리티 행렬을 평가하는 **시뮬레이션 및 채점 환경**을 포함하고 있습니다. 

단순히 교환 법칙만 확인하는 장난감 모델을 넘어, 구글의 초고속 양자 시뮬레이터인 `Stim`과 MWPM 디코더인 `PyMatching`을 활용하여 **실제 물리적 노이즈 환경에서의 논리적 에러율(Logical Error Rate)**을 엄격하게 측정합니다.

---

## 📂 핵심 파일: `stim_interface.py`

이 모듈의 핵심은 `StimEvaluator` 클래스입니다. 강화학습 에이전트가 생성한 `H_x`, `H_z` 행렬을 입력받아, 꼼수(Reward Hacking)가 통하지 않는 진정한 양자 코드로 평가합니다.

### 🎯 주요 평가 기준 (The Strict Judge)

1. **쌍대성 검증 (Dual Basis Evaluation):** - Z 에러(위상 뒤집힘)와 X 에러(비트 뒤집힘)를 모두 방어하는지 시뮬레이션합니다. 에이전트가 한쪽 방어에만 큐비트를 몰빵하는 것을 막기 위해, 둘 중 **가장 취약한 에러율(Worst-case)**을 최종 점수로 반환합니다.
2. **반교환성 필수 조건 (Anti-commutation Check):** - 유효한 양자 코드가 되기 위한 필수 조건인 서로 반교환(Anti-commute)하는 논리 연산자 쌍(`L_X`, `L_Z`)이 존재하는지 GF(2) 선형대수학을 통해 엄밀히 검증합니다.
3. **현실적인 노이즈 주입 (Depolarizing Channel):** - 단순한 단일 에러 모델이 아닌, 데이터 큐비트에 X, Y, Z 에러가 모두 발생할 수 있는 `DEPOLARIZE1` 채널을 적용하여 가혹한 테스트 환경을 제공합니다.

---

## 🛠️ `StimEvaluator` 주요 메서드 구조

* **`evaluate_logical_error_rate(Hx, Hz, num_shots=10000)`**
  * **역할:** 에이전트의 보상(Value)을 결정하는 최종 채점 함수.
  * **프로세스:** 유효한 논리 연산자를 찾고, X/Z 기저에 대해 각각 1만 번의 샷(Shot)을 시뮬레이션한 뒤, 가장 높은 에러율을 반환합니다. 코드가 기형적일 경우 가차 없이 `1.0` (100% 에러)을 반환합니다.

* **`_find_logical_operators(Hx, Hz)`**
  * **역할:** H_x의 영공간과 H_z의 영공간을 분석하여, 안정자(Stabilizer)에 속하지 않으면서 서로 내적했을 때 홀수(반교환)가 되는 완벽한 `L_X`, `L_Z` 쌍을 찾아냅니다.

* **`_simulate_code_capacity(H_measure, L_obs, noise_prob, is_x_basis)`**
  * **역할:** Stim 회로(Circuit)를 동적으로 생성하는 엔진입니다.
  * **프로세스:** 큐비트 초기화 ➔ 노이즈 주입 ➔ 신드롬 측정(C-NOT 게이트 기반) ➔ 데이터 큐비트 최종 측정 ➔ PyMatching을 통한 디코딩 및 실제 에러율 계산.

* **`save_circuit_diagram(Hx, Hz, save_dir, filename)`**
  * **역할:** 시각화 및 MLOps 유틸리티.
  * **설명:** 최종 평가를 통과한 '기적의 코드'에 대하여, 논문에 바로 삽입할 수 있는 수준의 깔끔한 타임라인 회로도(SVG)를 지정된 폴더에 저장합니다.

---

## 🚀 사용 예시 (Usage)

```python
from sim.stim_interface import StimEvaluator

# 1. 7큐비트, 물리적 에러율 1% 환경의 심판 생성
evaluator = StimEvaluator(num_qubits=7, noise_rate=0.01)

# 2. 에이전트가 만든 Hx, Hz 평가
# (AI가 제대로 된 코드를 만들었다면 0.01 미만의 값이 반환됨)
logical_error = evaluator.evaluate_logical_error_rate(Hx, Hz)
print(f"최종 논리적 에러율: {logical_error:.4f}")

# 3. 최고 성능 달성 시 회로도 박제
evaluator.save_circuit_diagram(Hx, Hz, save_dir="outputs/best_codes")