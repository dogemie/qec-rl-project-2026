import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools

class QECEnv(gym.Env):
    """
    GFlowNet 기반 QEC(Quantum Error Correction) 코드 생성을 위한 커스텀 환경.
    """
    def __init__(self, num_qubits=9, num_stabilizers=4, max_edges=32, max_weight=4, evaluator=None):
        super(QECEnv, self).__init__()
        
        self.n = num_qubits
        self.m = num_stabilizers
        self.max_edges = max_edges
        self.max_weight = max_weight
        self.evaluator = evaluator
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, self.m, self.n), dtype=np.int8
        )
        
        # 🌟 [핵심 변경 1] Action Space 1개 추가 (+1)
        # 기존: 2 * m * n (모두 선 긋기 액션)
        # 변경: 2 * m * n + 1 (마지막 액션은 '제출(Submit)' 버튼으로 사용)
        self.num_edge_actions = 2 * self.m * self.n
        self.submit_action = self.num_edge_actions
        self.action_space = spaces.Discrete(self.num_edge_actions + 1)
        
        self.state = None
        self.current_step = 0
        
        self.valid_topology_mask = self._build_topology_mask()

    def _build_topology_mask(self):
        # (이전과 동일)
        mask = np.zeros((2, self.m, self.n), dtype=np.int8)
        n_grid = int(np.ceil(np.sqrt(self.n)))
        grid_size = n_grid * 2 + 5
        center = grid_size // 2
        
        if center % 2 != 0: center += 1
            
        data_coords, x_stab_coords, z_stab_coords = [], [], []
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
        super().reset(seed=seed)
        self.state = np.zeros((2, self.m, self.n), dtype=np.int8)
        self.current_step = 0
        return self.state.copy(), self._get_info()

    def step(self, action):
        """
        GFlowNet 규칙에 따라 중간 보상 없이 상태를 전진(Forward)시킵니다.
        """
        # 🌟 [핵심 변경 2] '제출' 액션 처리
        if action == self.submit_action:
            terminated = True
            reward = self._get_final_reward()
            return self.state.copy(), reward, terminated, False, self._get_info()
            
        # 선 긋기 액션 처리
        c = action // (self.m * self.n)
        rem = action % (self.m * self.n)
        r = rem // self.n
        col = rem % self.n
        
        self.state[c, r, col] = 1
        self.current_step += 1
        
        # 🌟 [핵심 변경 3] 중간 보상 삭제 (GFlowNet은 도중에 점수를 주지 않습니다)
        reward = 0.0 
        
        terminated = False
        truncated = False
        if self.current_step >= self.max_edges:
            terminated = True
            reward = self._get_final_reward()
            
        return self.state.copy(), reward, terminated, truncated, self._get_info()

    def get_valid_action_mask(self):
        """
        오류가 있던 마스크 로직을 깔끔하게 수정했습니다.
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        # 1. 선 긋기 액션 (0 ~ num_edge_actions - 1)
        for action in range(self.num_edge_actions):
            c = action // (self.m * self.n)
            rem = action // self.n
            r = (action % (self.m * self.n)) // self.n
            col = action % self.n
            
            # 토폴로지 상 불가
            if self.valid_topology_mask[c, r, col] == 0: continue
            
            # 이미 선이 그어진 곳은 두 번 그을 수 없음 (DAG 조건 만족)
            if self.state[c, r, col] == 1: continue
            
            # 가중치 초과 방지
            if np.sum(self.state[c, r, :]) >= self.max_weight: continue
            
            mask[action] = 1
            
        # 2. 제출 액션 (마지막 인덱스)은 언제나 선택 가능하도록 열어둡니다.
        # (원한다면 최소 1개 이상 선을 그었을 때만 활성화할 수도 있습니다)
        mask[self.submit_action] = 1
        
        return mask

    def _get_info(self):
        return {
            "action_mask": self.get_valid_action_mask(),
            "current_step": self.current_step
        }
    
    def _get_final_reward(self):
        if self.evaluator is None: return 0.0
        from train.module.reward_calculator import calculate_qec_reward
        Hx, Hz = self.state[0], self.state[1]
        reward_res = calculate_qec_reward(Hx, Hz, self.n, self.m, self.evaluator)
        return reward_res["final_value"]

    # 🌟 [GFlowNet 전용 보너스 함수]
    def get_uniform_backward_prob(self):
        """
        현재 상태에서 역방향(Backward)으로 갈 수 있는 확률(P_B)을 반환합니다.
        현재 그어진 선의 개수가 N개라면, 직전 상태에서 올 수 있는 경우의 수는 N개입니다.
        """
        if self.current_step == 0:
            return 1.0
        return 1.0 / self.current_step