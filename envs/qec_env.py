import numpy as np
import gymnasium as gym
from gymnasium import spaces

class QECEnv(gym.Env):
    """
    QEC (Quantum Error Correction) 탐색을 위한 MCTS/RL 커스텀 환경
    """
    def __init__(self, num_qubits=7, num_stabilizers=3, max_edges=18, max_weight=4):
        super(QECEnv, self).__init__()
        
        self.n = num_qubits
        self.m = num_stabilizers
        self.max_edges = max_edges  # 한 에피소드당 허용되는 최대 행동(1을 배치하는) 횟수
        self.max_weight = max_weight # 각 안정자(행)가 가질 수 있는 최대 가중치(1의 개수)
        
        # 상태 공간: [2, m, n] 형태의 이진 텐서 (채널 0: H_X, 채널 1: H_Z)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, self.m, self.n), dtype=np.int8
        )
        
        # 행동 공간: 2 * m * n 개의 위치 중 하나를 선택하는 1D 이산(Discrete) 공간
        self.action_space = spaces.Discrete(2 * self.m * self.n)
        
        self.state = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """에피소드를 초기화합니다."""
        super().reset(seed=seed)
        self.state = np.zeros((2, self.m, self.n), dtype=np.int8)
        self.current_step = 0
        
        # 관측값(상태)과 부가 정보(info) 반환
        return self.state.copy(), self._get_info()

    def step(self, action):
        """에이전트가 행동을 취했을 때 상태를 변경하고 보상을 계산합니다."""
        # 1. 1D 행동 인덱스를 3D 좌표 (channel, row, col)로 디코딩
        c = action // (self.m * self.n)
        rem = action % (self.m * self.n)
        r = rem // self.n
        col = rem % self.n
        
        # 2. 상태 업데이트 (해당 위치의 비트를 1로 변경)
        self.state[c, r, col] = 1
        self.current_step += 1
        
        # 3. 중간 보상(Dense Reward) 계산
        reward = self._calculate_step_reward(c, r)
        
        # 4. 종료 조건 확인
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_edges:
            terminated = True
            # TODO: 종료 시점에 Stim 시뮬레이터를 호출하여 최종 에러율 기반의 Sparse Reward 부여
            # final_reward = self._get_stim_logical_error_reward()
            # reward += final_reward
            
        return self.state.copy(), reward, terminated, truncated, self._get_info()

    def _calculate_step_reward(self, changed_c, changed_r):
        """
        희소 보상 문제를 해결하기 위한 중간 보상 시스템.
        """
        reward = 0.0
        
        # 규칙 1. 가중치 패널티 (Weight Penalty)
        # 특정 안정자(행)에 1이 너무 많으면 에러가 증폭되므로 강한 패널티 부여
        row_weight = np.sum(self.state[changed_c, changed_r, :])
        if row_weight > self.max_weight:
            reward -= 1.0  # 타겟 가중치를 넘어가면 감점
        elif row_weight == self.max_weight:
            reward += 0.2  # 정확히 타겟 가중치에 도달하면 칭찬
            
        # 규칙 2. 선형 독립성 보상 (Rank Reward) - 옵션
        # 행렬의 랭크가 증가했다면 의미 있는 안정자를 추가한 것이므로 추가 점수
        # (계산 오버헤드를 줄이기 위해 필요에 따라 생략 가능)
        
        return reward

    def get_valid_action_mask(self):
        """MCTS 트리 탐색 시 불가능한 행동을 마스킹합니다."""
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        for action in range(self.action_space.n):
            c = action // (self.m * self.n)
            rem = action % (self.m * self.n)
            r = rem // self.n
            col = rem % self.n
            
            # 1. 이미 1인 자리는 다시 둘 수 없음
            if self.state[c, r, col] == 1:
                continue
                
            # 2. 한 행에 1이 너무 많아지는 것(max_weight 초과)을 물리적으로 차단
            if np.sum(self.state[c, r, :]) >= self.max_weight:
                continue
            
            # 임시적인 교환 법칙 위반은 허용! (나중에 최종 완성본에서 검사)
            mask[action] = 1 
                
        return mask

    def _get_info(self):
        """디버깅 및 로깅을 위한 추가 정보"""
        return {
            "action_mask": self.get_valid_action_mask(),
            "current_step": self.current_step
        }