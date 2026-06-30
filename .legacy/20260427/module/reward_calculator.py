import numpy as np

def calculate_qec_reward(Hx, Hz, num_qubits, num_stabilizers, evaluator, beta=3.0):
    """
    GFlowNet을 위한 엄격한 양수(Strictly Positive) 보상 함수.
    결함이 많을수록 0에 수렴하는 극소의 양수를, 완벽하면 1.0 이상의 큰 양수를 반환합니다.
    """
    dot_product = np.dot(Hx, Hz.T)
    violations = np.sum((dot_product % 2) != 0)
    
    orphaned_x = np.sum(np.sum(Hx, axis=0) == 0)
    orphaned_z = np.sum(np.sum(Hz, axis=0) == 0)
    total_orphans = orphaned_x + orphaned_z
    
    overlap_X = np.dot(Hx, Hx.T)
    np.fill_diagonal(overlap_X, 0)
    girth4_X = np.sum(overlap_X >= 2) // 2 
    
    overlap_Z = np.dot(Hz, Hz.T)
    np.fill_diagonal(overlap_Z, 0)
    girth4_Z = np.sum(overlap_Z >= 2) // 2
    
    total_girth4_cycles = girth4_X + girth4_Z
    
    result = {
        "final_value": 0.0, # 계산 후 반드시 0 초과의 양수로 덮어씌워짐
        "err_01": 1.0,
        "err_001": 1.0,
        "distance": -1,
        "cnots": -1,
        "violations": violations,
        "orphans": total_orphans,
        "is_valid": False
    }

    # 🌟 1. 결함(Defect) 스코어 계산
    # 각 결함의 치명도에 따라 가중치를 줍니다. (위반=2.0, 고아=0.5, Girth4=1.0)
    defect_score = (violations * 2.0) + (total_orphans * 0.5) + (total_girth4_cycles * 1.0)
    
    if defect_score > 0:
        # 🌟 2. 지수 감쇠(Exponential Decay) 적용
        # 결함이 1개라도 있으면 보상은 1.0 미만으로 뚝 떨어집니다.
        # 예: defect_score가 2.0이면, np.exp(-6.0) ≈ 0.0024 (매우 작은 양수)
        result["final_value"] = float(np.exp(-beta * defect_score))
        return result
        
    # =====================================================================
    # 🌟 3. 완벽한 코드 (결함 0개) 발견 시 잭팟 보상 부여
    # =====================================================================
    result["is_valid"] = True
    base_validity_reward = 1.0  # 결함이 0개일 때의 기본값 (e^0 = 1.0)
    
    # 에러율 평가 (병목 구간이므로 정답일 때만 실행)
    err_01 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate_x=0.01, noise_rate_z=0.01)
    err_001 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate_x=0.001, noise_rate_z=0.001)
    err_05 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate_x=0.05, noise_rate_z=0.05)
    
    result["err_01"] = err_01
    result["err_001"] = err_001
    
    total_cnots = np.sum(Hx) + np.sum(Hz)
    result["cnots"] = total_cnots
    result["distance"] = total_cnots * 2
    
    if err_01 >= 1.0 or err_001 >= 1.0 or err_05 >= 1.0:
        result["final_value"] = base_validity_reward
    else:
        # 방어력이 뛰어날수록 보상이 1.0을 뚫고 2.0, 2.5까지 올라갑니다!
        imp_01 = max(0, (0.01 - err_01) / 0.01)
        imp_001 = max(0, (0.001 - err_001) / 0.001)
        imp_05 = max(0, (0.05 - err_05) / 0.05)
        
        defense_score = np.clip((imp_01 * 0.5) + (imp_001 * 0.3) + (imp_05 * 0.2), 0.0, 1.0)
        
        # CNOT 개수가 적을수록 추가 보상
        sparsity_penalty = total_cnots / (num_qubits * num_stabilizers * 2)
        hw_score = (1.0 - sparsity_penalty) * 0.5
        
        final_val = base_validity_reward + defense_score + hw_score
        result["final_value"] = float(final_val)
        
    return result