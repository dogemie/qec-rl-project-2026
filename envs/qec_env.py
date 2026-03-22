import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools

class QECEnv(gym.Env):
    """
    QEC (Quantum Error Correction) 탐색을 위한 MCTS/RL 커스텀 환경.
    """
    def __init__(self, num_qubits=9, num_stabilizers=4, max_edges=32, max_weight=4, evaluator=None):
        """
        QECEnv 환경을 초기화합니다.
        
        Args:
            num_qubits (int): 시스템에 존재하는 데이터 큐비트의 개수.
            num_stabilizers (int): X 또는 Z 안정자 큐비트의 각각의 개수.
            max_edges (int): 한 에피소드당 허용되는 최대 행동(CNOT 배치) 횟수.
            max_weight (int): 각 안정자(행)가 가질 수 있는 최대 데이터 큐비트 연결 수.
        """
        super(QECEnv, self).__init__()
        
        self.n = num_qubits
        self.m = num_stabilizers
        self.max_edges = max_edges
        self.max_weight = max_weight
        self.evaluator = evaluator
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, self.m, self.n), dtype=np.int8
        )
        self.action_space = spaces.Discrete(2 * self.m * self.n)
        
        self.state = None
        self.current_step = 0
        
        # 하드웨어 구조에 따른 물리적 연결 가능 여부를 사전에 계산한 마스크
        self.valid_topology_mask = self._build_topology_mask()

    def _build_topology_mask(self):
        """
        하드웨어의 격자(Grid) 구조를 기반으로 물리적으로 인접한(Manhattan distance 2) 
        큐비트 쌍만 1로 활성화된 마스크 배열을 생성합니다.
        
        Returns:
            np.ndarray: [2, m, n] 형태의 이진 텐서 마스크.
        """
        mask = np.zeros((2, self.m, self.n), dtype=np.int8)
        
        n_grid = int(np.ceil(np.sqrt(self.n)))
        grid_size = n_grid * 2 + 5
        center = grid_size // 2
        
        if center % 2 != 0:
            center += 1
            
        data_coords = []
        x_stab_coords = []
        z_stab_coords = []
        
        for y in range(grid_size):
            for x in range(grid_size):
                dist = max(abs(x - center), abs(y - center))
                if x % 2 == 1 and y % 2 == 1:
                    data_coords.append((dist, y, x, x, -y))
                elif x % 2 == 0 and y % 2 == 0:
                    if (x // 2 + y // 2) % 2 == 0:
                        x_stab_coords.append((dist, y, x, x, -y))
                    else:
                        z_stab_coords.append((dist, y, x, x, -y))
                        
        data_coords.sort(); x_stab_coords.sort(); z_stab_coords.sort()
        data_coords = [(x, y) for _, _, _, x, y in data_coords]
        x_stab_coords = [(x, y) for _, _, _, x, y in x_stab_coords]
        z_stab_coords = [(x, y) for _, _, _, x, y in z_stab_coords]
        
        for c in range(2):
            stab_list = x_stab_coords if c == 0 else z_stab_coords
            for r in range(self.m):
                if r < len(stab_list): sx, sy = stab_list[r]
                else: continue
                for col in range(self.n):
                    if col < len(data_coords): dx, dy = data_coords[col]
                    else: continue
                    if abs(sx - dx) + abs(sy - dy) == 2:
                        mask[c, r, col] = 1
        return mask
    
    def reset(self, seed=None, options=None):
        """
        에피소드 상태를 초기화합니다.
        
        Args:
            seed (int, optional): 난수 시드.
            options (dict, optional): 추가 환경 옵션.
            
        Returns:
            tuple: 초기 상태 텐서와 추가 정보 딕셔너리.
        """
        super().reset(seed=seed)
        self.state = np.zeros((2, self.m, self.n), dtype=np.int8)
        self.current_step = 0
        return self.state.copy(), self._get_info()

    def step(self, action):
        """
        에이전트가 선택한 행동을 적용하여 상태를 갱신합니다.
        
        Args:
            action (int): 선택된 1D 행동 인덱스.
            
        Returns:
            tuple: (다음 상태, 보상, 종료 여부, 잘림 여부, 추가 정보)
        """
        c = action // (self.m * self.n)
        rem = action % (self.m * self.n)
        r = rem // self.n
        col = rem % self.n
        
        if self.state[c, r, col] == 1:
            terminated = True
            final_reward = self._get_final_reward()
            return self.state.copy(), final_reward, terminated, False, self._get_info()
            
        self.state[c, r, col] = 1
        self.current_step += 1
        
        reward = self._calculate_step_reward(c, r)
        
        terminated = False
        truncated = False
        if self.current_step >= self.max_edges:
            terminated = True
            reward += self._get_final_reward()
            
        return self.state.copy(), reward, terminated, truncated, self._get_info()

    def _calculate_step_reward(self, changed_c, changed_r):
        row_weight = np.sum(self.state[changed_c, changed_r, :])
        if row_weight > self.max_weight:
            return -1.0 
        return -0.01

    def get_valid_action_mask(self):
        """
        현재 상태와 하드웨어 토폴로지를 고려하여 유효한 행동 범위를 계산합니다.
        
        Returns:
            np.ndarray: 유효한 행동은 1, 그 외는 0으로 지정된 1D 마스크 배열.
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        for action in range(self.action_space.n):
            mask = np.zeros(self.action_space.n, dtype=np.int8)
        for action in range(self.action_space.n):
            c = action // (self.m * self.n)
            rem = action % (self.m * self.n)
            r = rem // self.n
            col = rem % self.n
            
            if self.valid_topology_mask[c, r, col] == 0: continue
            if self.state[c, r, col] == 1:
                mask[action] = 1
                continue
            if np.sum(self.state[c, r, :]) >= self.max_weight: continue
            mask[action] = 1
        return mask

    def _get_info(self):
        """
        현재 환경의 진행 상태 및 유효 행동 마스크 정보를 반환합니다.
        
        Returns:
            dict: 에이전트 탐색을 보조할 추가 정보.
        """
        return {
            "action_mask": self.get_valid_action_mask(),
            "current_step": self.current_step
        }
    
    def _get_final_reward(self):
        """
        에피소드 종료 시점에 최종 보상을 계산하기 위해 채점관(evaluator)을 호출합니다.
        
        Returns:
            float: 최종 보상값.
        """
        if self.evaluator is None:
            return 0.0
        
        from train.module.reward_calculator import calculate_qec_reward
        Hx, Hz = self.state[0], self.state[1]
        reward_res = calculate_qec_reward(Hx, Hz, self.n, self.m, self.evaluator)
        
        return reward_res["final_value"]