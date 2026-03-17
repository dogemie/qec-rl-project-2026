import numpy as np

def calculate_qec_reward(Hx, Hz, num_qubits, num_stabilizers, evaluator):
    """
    AI가 생성한 Hx, Hz 행렬을 평가하여 최종 가치(Value)와 각종 메타 데이터를 반환합니다.
    """
    dot_product = np.dot(Hx, Hz.T)
    
    # 1. 제약 조건 위반 검사
    violations = np.sum((dot_product % 2) != 0)
    orphaned_x = np.sum(np.sum(Hx, axis=0) == 0)
    orphaned_z = np.sum(np.sum(Hz, axis=0) == 0)
    total_orphans = orphaned_x + orphaned_z
    
    # 반환할 기본 딕셔너리 구조 세팅
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

    # 🌟 1. 유효하지 않은 코드 (위반 또는 버려진 큐비트 존재)
    if violations > 0 or total_orphans > 0:
        max_violations = num_stabilizers * num_stabilizers
        violation_ratio = violations / max_violations
        orphan_ratio = total_orphans / num_qubits
        
        # 지수적 보상 셰이핑 (Exponential Reward Shaping)
        penalty_score = (violation_ratio ** 3) * 0.8 + (orphan_ratio ** 2) * 0.2
        result["final_value"] = -1.0 * penalty_score
        
        # [희망고문] 위반이 1~2개로 매우 적을 때는 약간의 플러스 점수를 주어 유도함
        if violations <= 2 and total_orphans == 0:
             result["final_value"] = 0.2 
             
        return result

    # 🌟 2. [돌파구!] 유효한 코드 발견 (위반 0, 고아 0)
    result["is_valid"] = True
    base_validity_reward = 0.5 
    
    # 에러율 평가 (Stim)
    err_01 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate=0.01)
    err_001 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate=0.001)
    err_05 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate=0.05)
    
    result["err_01"] = err_01
    result["err_001"] = err_001
    
    # 2D 기하학적 거리 패널티 (범용 동적 격자 확장 및 패리티 매핑)
    total_nodes = num_qubits + num_stabilizers * 2
    n = int(np.ceil(np.sqrt(num_qubits)))
    grid_size = n * 2 + 5
    
    def get_coord(index):
        if not hasattr(get_coord, "mapping"):
            mapping = {}
            center = grid_size // 2
            data_coords = []
            stab_coords = []
            
            for y in range(grid_size):
                for x in range(grid_size):
                    dist = max(abs(x - center), abs(y - center))
                    if x % 2 == 1 and y % 2 == 1:
                        data_coords.append((dist, y, x, x, -y))
                    else:
                        stab_coords.append((dist, y, x, x, -y))
                        
            data_coords.sort()
            stab_coords.sort()
            
            data_coords = [(x, y) for _, _, _, x, y in data_coords]
            stab_coords = [(x, y) for _, _, _, x, y in stab_coords]
            
            for i in range(total_nodes):
                if i < num_qubits:
                    mapping[i] = data_coords.pop(0) if data_coords else stab_coords.pop(0)
                else:
                    mapping[i] = stab_coords.pop(0) if stab_coords else data_coords.pop(0)
            get_coord.mapping = mapping

        return get_coord.mapping[index]
    
    # 거리 및 CNOT 계산
    total_distance = 0
    for i in range(Hx.shape[0]):
        stab_coord = get_coord(num_qubits + i)
        for j in range(Hx.shape[1]):
            if Hx[i, j] == 1:
                qubit_coord = get_coord(j)
                total_distance += abs(stab_coord[0] - qubit_coord[0]) + abs(stab_coord[1] - qubit_coord[1])
    
    for i in range(Hz.shape[0]):
        stab_coord = get_coord(num_qubits + num_stabilizers + i)
        for j in range(Hz.shape[1]):
            if Hz[i, j] == 1:
                qubit_coord = get_coord(j)
                total_distance += abs(stab_coord[0] - qubit_coord[0]) + abs(stab_coord[1] - qubit_coord[1])
    
    total_cnots = np.sum(Hx) + np.sum(Hz)
    result["distance"] = total_distance
    result["cnots"] = total_cnots
    
    # 가치 점수 종합 계산
    if err_01 >= 1.0 or err_001 >= 1.0 or err_05 >= 1.0:
        result["final_value"] = base_validity_reward
    else:
        imp_01 = max(0, (0.01 - err_01) / 0.01)
        imp_001 = max(0, (0.001 - err_001) / 0.001)
        imp_05 = max(0, (0.05 - err_05) / 0.05)
        
        defense_score = np.clip((imp_01 * 0.5) + (imp_001 * 0.3) + (imp_05 * 0.2), 0.0, 0.5)
        
        distance_penalty = min(1.0, total_distance / (total_cnots * grid_size))
        sparsity_penalty = total_cnots / (num_qubits * num_stabilizers * 2)
        
        hw_score = ((1.0 - distance_penalty) * 0.5) + ((1.0 - sparsity_penalty) * 0.5)
        hw_score = hw_score * 0.5 
        
        final_val = base_validity_reward + (defense_score * 0.5) + hw_score
        result["final_value"] = np.clip(final_val, 0.0, 1.0)
        
    return result