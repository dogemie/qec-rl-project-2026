# envs/qldpc_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class QLDPCEnv(gym.Env):
    """
    차세대 QLDPC / Bivariate Bicycle(BB) 코드를 찾기 위한 다항식 시프트(Shift) 탐색 환경.
    개별 픽셀(CNOT)이 아닌, 블록 단위의 행렬 이동을 학습합니다.
    """
    def __init__(self, block_size=6, num_shifts=4):
        super(QLDPCEnv, self).__init__()
        
        self.L = block_size  # 순환 행렬(Circulant Matrix)의 크기 (예: 6x6)
        self.num_shifts = num_shifts # 찾아야 할 다항식 시프트 값의 개수
        
        # 🌟 상태(State): 수많은 0과 1의 격자가 아니라, 단순히 시프트 값들의 배열입니다!
        # 예: [1, 3, 0, 5] -> A_1 블록은 1칸 시프트, B_1 블록은 3칸 시프트...
        self.observation_space = spaces.MultiDiscrete([self.L] * self.num_shifts)
        
        # 🌟 행동(Action): 특정 시프트 인덱스를 선택하여 값을 +1 하거나 -1 하는 행동
        # 탐색 공간이 우주적 규모에서 동네 놀이터 규모로 극단적으로 압축됩니다.
        self.action_space = spaces.Discrete(self.num_shifts * 2)
        
        self.state = np.zeros(self.num_shifts, dtype=np.int8)
        self.current_step = 0
        self.max_steps = 20 # 시프트를 조절할 수 있는 턴 수

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 초기 상태를 무작위 시프트 배열로 시작하거나 0으로 시작
        self.state = np.zeros(self.num_shifts, dtype=np.int8)
        self.current_step = 0
        return self.state.copy(), {}

    def step(self, action):
        # 1. Action 디코딩 (어떤 시프트 값을, 어떻게 조절할 것인가?)
        shift_idx = action // 2
        direction = 1 if action % 2 == 0 else -1
        
        # 2. 상태 업데이트 (순환 행렬이므로 범위를 L로 나눈 나머지로 유지)
        self.state[shift_idx] = (self.state[shift_idx] + direction) % self.L
        self.current_step += 1
        
        # 3. 종료 조건 및 보상 계산
        terminated = self.current_step >= self.max_steps
        reward = 0.0
        
        if terminated:
            # TODO: self.state 배열 값들로 Hx, Hz 순환 블록 행렬을 조립한 뒤, 
            # StimEvaluator를 호출해 실제 논리적 에러율(P_L)을 보상으로 반환합니다.
            pass
            
        return self.state.copy(), reward, terminated, False, {}