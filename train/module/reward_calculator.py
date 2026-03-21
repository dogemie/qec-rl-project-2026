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

    # 교환 법칙이 망가진 정도에 따라 패널티를 다르게
    if violations > 0 or total_orphans > 0:
        
        # 1. 패널티 계산 (위반은 엄격하게 깎음)
        violation_penalty = -0.2 * violations
        
        # 2. 긍정적 보상 계산 (고아가 적을수록, 즉 연결을 많이 할수록 플러스 점수)
        # 전체 큐비트 중 '연결된 큐비트의 비율'을 점수로 줍니다. (0.0 ~ +0.4)
        total_nodes = num_qubits + num_stabilizers * 2 # 예시
        connected_ratio = 1.0 - (total_orphans / total_nodes)
        connection_reward = 0.4 * connected_ratio
        
        # 3. 중간 가치(Value) 합산
        # 아무것도 안 한 빈 판은 0점 근처, 위반을 내면 마이너스, 위반 없이 연결을 늘려가면 플러스가 됩니다.
        intermediate_value = violation_penalty + connection_reward
        
        # 4. 완성된 코드(Valid)에 대한 엄청난 잭팟 보상
        if violations == 0 and total_orphans == 0:
            result["is_valid"] = True
            base_validity_reward = 0.5
            
            # ... (이후 방어력 점수(err_01 등), 하드웨어 점수(distance) 계산 로직 동일)
            
            final_val = base_validity_reward + (defense_score * 0.5) + hw_score
            result["final_value"] = np.clip(final_val, 0.0, 1.0)
        else:
            # 미완성 상태면 위에서 계산한 중간 가치를 반환 (최소 -1.0 최대 0.4)
            result["final_value"] = max(-1.0, intermediate_value)

        return result

    result["is_valid"] = True
    base_validity_reward = 0.5 
    
    # X, Z 기저 모두에서 3가지 노이즈 레벨에 대한 에러율을 평가하여 방어력 점수 계산에 활용
    err_01 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate_x=0.01, noise_rate_z=0.01)
    err_001 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate_x=0.001, noise_rate_z=0.001)
    err_05 = evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate_x=0.05, noise_rate_z=0.05)
    
    result["err_01"] = err_01
    result["err_001"] = err_001
    
    total_nodes = num_qubits + num_stabilizers * 2
    n = int(np.ceil(np.sqrt(num_qubits)))
    grid_size = n * 2 + 5
    
    def get_coord(index):
        if not hasattr(get_coord, "mapping"):
            mapping = {}
            center = grid_size // 2
            
            # 중심점 짝수/홀수 보정 (center가 항상 짝수가 되도록 맞춤)
            if center % 2 != 0:
                center += 1

            data_coords = []
            stab_coords = []
            
            # 🌟 PDF 기반 완벽한 (2n, 2n+1) 규칙 적용
            for y in range(grid_size):
                for x in range(grid_size):
                    dist = max(abs(x - center), abs(y - center))
                    
                    if x % 2 == 1 and y % 2 == 1:
                        # 데이터 큐비트: 무조건 (홀수, 홀수)
                        data_coords.append((dist, y, x, x, -y))
                    elif x % 2 == 0 and y % 2 == 0:
                        # 안정자: 무조건 (짝수, 짝수)
                        stab_coords.append((dist, y, x, x, -y))
                        
            data_coords.sort()
            stab_coords.sort()
            
            data_coords = [(x, y) for _, _, _, x, y in data_coords]
            stab_coords = [(x, y) for _, _, _, x, y in stab_coords]
            
            for i in range(total_nodes):
                if i < num_qubits:
                    mapping[i] = data_coords.pop(0)
                else:
                    mapping[i] = stab_coords.pop(0)
            get_coord.mapping = mapping

        return get_coord.mapping[index]
    
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
    
    if err_01 >= 1.0 or err_001 >= 1.0 or err_05 >= 1.0:
        result["final_value"] = base_validity_reward
    else:
        imp_01 = max(0, (0.01 - err_01) / 0.01)
        imp_001 = max(0, (0.001 - err_001) / 0.001)
        imp_05 = max(0, (0.05 - err_05) / 0.05)
        
        defense_score = np.clip((imp_01 * 0.5) + (imp_001 * 0.3) + (imp_05 * 0.2), 0.0, 0.5)
        
        # 🌟 (중요) Rotated Grid에서는 가장 짧은 CNOT 연결 거리가 2입니다.
        # 따라서 총 최소 거리는 (total_cnots * 2)가 됩니다.
        optimal_distance = total_cnots * 2
        distance_penalty = min(1.0, max(0, total_distance - optimal_distance) / (total_cnots * 2))
        
        sparsity_penalty = total_cnots / (num_qubits * num_stabilizers * 2)
        
        hw_score = ((1.0 - distance_penalty) * 0.5) + ((1.0 - sparsity_penalty) * 0.5)
        hw_score = hw_score * 0.5 
        
        final_val = base_validity_reward + (defense_score * 0.5) + hw_score
        result["final_value"] = np.clip(final_val, 0.0, 1.0)
        
    return result