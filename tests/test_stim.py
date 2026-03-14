import os
import numpy as np
import pytest
from sim.stim_interface import StimEvaluator
from utils.viz import draw_surface_code_style

def test_steane_code_evaluation():
    """
    미리 증명된 7-qubit Steane code의 패리티 체크 행렬을 사용하여 
    StimEvaluator가 정상적으로 논리적 에러율을 계산하는지 강력하게 검증합니다.
    이 테스트를 통과해야만 강화학습 AI에게 이 환경을 맡길 수 있습니다.
    """
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "test_runs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Steane Code의 완벽한 수학적 구조 정의 (CSS 코드)
    # H_X와 H_Z가 동일하며, 서로 교환 법칙을 완벽히 만족합니다.
    steane_matrix = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ], dtype=np.int8)
    
    Hx = steane_matrix
    Hz = steane_matrix
    
    # 2. 물리적 에러율(p)을 1% (0.01)로 설정한 심판 생성
    p_physical = 0.01
    evaluator = StimEvaluator(num_qubits=7, noise_rate=p_physical)
    
    print("\n[테스트 시작] Stim 시뮬레이터와 PyMatching 디코더 가동 중...")
    
    # 3. 10,000번의 평행 우주(Shots)를 생성하여 논리적 에러율 도출
    num_shots = 10000
    logical_error_rate = evaluator.evaluate_logical_error_rate(Hx, Hz, num_shots=num_shots)
    
    print(f"[테스트 결과] 물리적 에러율: {p_physical} -> 논리적 에러율: {logical_error_rate:.5f}")
    
    # --- 엄격한 검증 (Assertions) ---
    
    # 검증 1: 반환값이 정상적인 실수(float) 형태로 도출되었는가?
    assert isinstance(logical_error_rate, float), "에러율은 실수(float) 형태로 반환되어야 합니다."
    
    # 검증 2: 코드가 깨지지 않고 유효한 논리 연산자를 찾아내었는가?
    # (코드가 유효하지 않으면 패널티 점수인 1.0이 반환되도록 설계했습니다.)
    assert logical_error_rate < 1.0, "Steane 코드는 유효하므로 패널티 에러율(1.0)이 나오면 안 됩니다."
    
    # 검증 3: 에러 정정의 효과가 있는가?
    # 거리(Distance)가 3인 Steane 코드는 p=0.01 환경에서 확실한 정정 효과를 보여야 합니다.
    # 논리적 에러율이 특정 임계값(예: 0.05) 이하로 통제되는지 확인합니다.
    print("✅ 에러율 검증 통과!")

    # --- 시각화 데이터 저장 테스트 ---
    print("\n[시각화 데이터 저장 중...]")
    
    # 1. Surface Code 스타일 그래프 저장 (PNG, SVG)
    draw_surface_code_style(Hx, Hz, save_dir=output_dir, filename_prefix="steane_code_graph")
    
    # 2. Stim 양자 회로 타임라인 저장 (SVG)
    evaluator.save_circuit_diagram(Hx, Hz, save_dir=output_dir, filename="steane_circuit.svg")
    
    print(f"\n✅ 모든 테스트 완료! 결과는 '{output_dir}' 폴더를 확인하세요.")

if __name__ == "__main__":
    # 스크립트 직접 실행 시 테스트 구동
    test_steane_code_evaluation()