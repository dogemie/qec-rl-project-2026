import numpy as np

def calculate_qec_reward(Hx, Hz, num_qubits, num_stabilizers, evaluator):
    """
    AI가 생성한 Hx, Hz 행렬을 평가하여 최종 가치(Value)와 각종 메타 데이터를 반환합니다.
    """
    dot_product = np.dot(Hx, Hz.T)
    
    violations = np.sum((dot_product % 2) != 0)
    orphaned_x = np.sum(np.sum(Hx, axis=0) == 0)
    orphaned_z = np.sum(np.sum(Hz, axis=0) == 0)
    total_orphans = orphaned_x + orphaned_z
    
    result = {
        "final_value": 0.0,
        "err_01": 1.0,
        "err_001": 1.0,
        "distance": -1,
        "cnots": -1,
        "violations": violations,
        "orphans": total_orphans,
        "is_valid": False
    }

    # 🌟 [NEW] Girth-4 (짧은 순환 고리) 탐지 연산
    # Hx 내적: 서로 다른 X 안정자가 몇 개의 데이터 큐비트를 공유하는지 계산
    overlap_X = np.dot(Hx, Hx.T)
    np.fill_diagonal(overlap_X, 0) # 자기 자신과의 공유는 무시
    # 2개 이상의 큐비트를 공유한다면 Girth-4 사이클이 존재한다는 뜻 (대칭행렬이므로 2로 나눔)
    girth4_X = np.sum(overlap_X >= 2) // 2 
    
    # Hz 내적: 서로 다른 Z 안정자가 몇 개의 데이터 큐비트를 공유하는지 계산
    overlap_Z = np.dot(Hz, Hz.T)
    np.fill_diagonal(overlap_Z, 0)
    girth4_Z = np.sum(overlap_Z >= 2) // 2
    
    total_girth4_cycles = girth4_X + girth4_Z

    # 교환 법칙 위반, 고아 큐비트, 또는 Girth-4 사이클이 하나라도 있으면 패널티 부과!
    if violations > 0 or total_orphans > 0 or total_girth4_cycles > 0:
        
        # 1. 교환 법칙 위반 패널티 (-0.2 per violation)
        violation_penalty = -0.2 * violations
        
        # 2. 고아 큐비트 패널티 (-0.05 per orphan)
        orphan_penalty = -0.05 * total_orphans
        
        # 3. 🌟 Girth-4 사이클 패널티 (-0.2 per cycle)
        # 겹치는 선이 많을수록 에러가 증폭되므로 강력한 철퇴를 내립니다.
        girth4_penalty = -0.2 * total_girth4_cycles
        
        # 4. 최종 패널티 합산 (안정성을 위해 최하점은 -1.0으로 방어)
        total_penalty = violation_penalty + orphan_penalty + girth4_penalty
        total_penalty = max(total_penalty, -1.0)
        
        result["final_value"] = total_penalty
        
        return result

    # =====================================================================
    # 위반(Violation)이 0개이고, 고아도 0개이며, Girth-4 사이클도 없는 
    # 완벽한 코드를 찾았을 때만 아래의 '논리적 에러율(Sinter) 평가'를 진행합니다.
    # =====================================================================

    result["is_valid"] = True
    base_validity_reward = 0.5 
    
    # 에러율 평가 (가장 시간이 오래 걸리는 병목 구간이므로, 완벽한 코드일 때만 실행)
    err_01 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate_x=0.01, noise_rate_z=0.01)
    err_001 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate_x=0.001, noise_rate_z=0.001)
    err_05 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate_x=0.05, noise_rate_z=0.05)
    
    result["err_01"] = err_01
    result["err_001"] = err_001
    
    total_cnots = np.sum(Hx) + np.sum(Hz)
    result["cnots"] = total_cnots
    
    # 하드웨어 토폴로지 마스킹 덕분에 모든 선의 거리는 '2'로 고정됩니다. 
    # 따라서 복잡한 좌표 계산 코드는 삭제하고, 단순 곱셈으로 대체합니다.
    optimal_distance = total_cnots * 2
    result["distance"] = optimal_distance
    
    # 방어력 점수 계산 로직
    if err_01 >= 1.0 or err_001 >= 1.0 or err_05 >= 1.0:
        result["final_value"] = base_validity_reward
    else:
        imp_01 = max(0, (0.01 - err_01) / 0.01)
        imp_001 = max(0, (0.001 - err_001) / 0.001)
        imp_05 = max(0, (0.05 - err_05) / 0.05)
        
        defense_score = np.clip((imp_01 * 0.5) + (imp_001 * 0.3) + (imp_05 * 0.2), 0.0, 0.5)
        
        # CNOT 개수가 적을수록(Sparsity) 좋은 하드웨어 점수를 줌
        sparsity_penalty = total_cnots / (num_qubits * num_stabilizers * 2)
        hw_score = ((1.0 - sparsity_penalty) * 0.5)
        
        final_val = base_validity_reward + (defense_score * 0.5) + hw_score
        result["final_value"] = np.clip(final_val, 0.0, 1.0)
        
    return result