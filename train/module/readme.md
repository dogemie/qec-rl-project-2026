# 🚂 Training Module (`train/module/`)

이 디렉토리는 AlphaZero 기반의 양자 오류 정정(QEC) 코드 탐색을 총괄하는 **핵심 학습 오케스트레이션(Orchestration) 모듈**을 포함하고 있습니다.

단순한 스크립트를 넘어, 실험의 재현성(Reproducibility)과 MLOps 파이프라인이 결합된 프로덕션 레벨의 훈련 시스템을 제공합니다.

## 📂 핵심 파일: `trainer.py`

이 모듈의 핵심은 `AlphaZeroTrainer` 클래스입니다. 에이전트(Agent), 환경(Environment), 시뮬레이터(Stim)를 하나로 묶어 실제 강화학습의 루프(Self-play & Training)를 구동합니다.

### 🧠 주요 역할 및 기능 (Key Features)

1. **AlphaZero 학습 루프 구동:**
   * MCTS(Monte Carlo Tree Search)를 통한 자가 대국(Self-play) 에피소드 생성.
   * 생성된 데이터를 바탕으로 신경망(Policy & Value) 학습 및 가중치 업데이트.
   * 학습 안정성을 위한 학습률 스케줄러(StepLR) 적용.

2. **양자 물리적 보상 체계 (QEC Reward Shaping):**
   * **기본 제약 검증:** 교환 법칙 위반 및 고립된(버려진) 물리 큐비트 발생 시 강력한 패널티 부여.
   * **논리 에러율 평가:** `StimEvaluator`를 호출하여 실제 물리적 노이즈(Depolarizing) 환경에서의 에러 정정 능력 평가 (베이스라인 $p=0.01$ 돌파 시 보상).
   * **희소성(Sparsity) 보상:** qLDPC 코드의 핵심 원리인 '선 밀도 최소화'를 유도하여 CNOT 게이트로 인한 에러 전파를 방지하는 범용적 보상 체계 적용.

3. **MLOps 및 실험 관리:**
   * **Seed 관리:** 난수 시드를 통한 완벽한 실험 재현성 보장 및 시드 자동 할당 기능.
   * **진화 히스토리 추적:** 최고 성능(에러율 갱신) 달성 시 덮어쓰지 않고 `best_codes_epc{}_ep{}` 형태로 중간 결과물 박제.
   * **시각화 자동화:** 최고 성능 코드의 행렬 데이터(.npy), 태너 그래프(Tanner Graph), Stim 회로도(SVG) 자동 저장.

---

**💡 참고:** 이 모듈은 직접 실행되지 않으며, 상위 폴더의 `train.py` 스크립트를 통해 인자(Arguments)를 전달받아 실행됩니다.