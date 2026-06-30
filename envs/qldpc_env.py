import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils.bb_code_builder import BBCodeBuilder

class QLDPCEnv(gym.Env):
    """
    GFlowNet을 위한 Bivariate Bicycle (BB) 코드 생성 환경.
    AI는 4개의 시프트(Shift) 값을 순차적으로 선택하며 H_X, H_Z 행렬을 완성합니다.
    """
    def __init__(self, L=15, evaluator=None):
        super(QLDPCEnv, self).__init__()
        
        self.L = L
        self.evaluator = evaluator
        self.builder = BBCodeBuilder(L)
        
        # 🌟 상태 공간 (State Space)
        # 4개의 빈칸을 가진 1D 배열. -1은 '아직 선택되지 않음'을 의미합니다.
        # 형태: [a1, a2, b1, b2]
        self.observation_space = spaces.Box(
            low=-1, high=L-1, shape=(4,), dtype=np.int32
        )
        
        # 🌟 행동 공간 (Action Space)
        # 0 ~ L-1: 시프트 숫자 선택
        # L (마지막 인덱스): '제출(Submit)' 버튼
        self.submit_action = self.L
        self.action_space = spaces.Discrete(self.L + 1)
        
        self.state = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([-1, -1, -1, -1], dtype=np.int32)
        self.current_step = 0
        return self.state.copy(), self._get_info()

    def step(self, action):
        """AI가 숫자를 하나 고르거나, 제출합니다."""
        
        # 1. 제출(Submit) 행동 처리
        if action == self.submit_action:
            terminated = True
            reward = self._get_final_reward()
            return self.state.copy(), reward, terminated, False, self._get_info()
            
        # 2. 숫자(Shift) 채워넣기 행동 처리
        self.state[self.current_step] = action
        self.current_step += 1
        
        # 중간 보상은 무조건 0 (GFlowNet 규칙)
        reward = 0.0
        terminated = False
        
        # 4칸이 모두 찼다면, 다음 턴에는 무조건 '제출'을 해야 함 (Mask로 강제됨)
        return self.state.copy(), reward, terminated, False, self._get_info()

    def get_valid_action_mask(self):
        """현재 턴에 할 수 있는 행동을 제한합니다 (Action Masking)"""
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        if self.current_step < 4:
            # 아직 4칸을 다 못 채웠다면, 0 ~ L-1 의 숫자 중 하나를 고를 수 있음
            mask[0:self.L] = 1
            # (선택적) a1 < a2, b1 < b2가 되도록 강제하면 중복 탐색(예: 1,3 과 3,1)을 절반으로 줄일 수 있습니다!
            if self.current_step in [1, 3]:
                prev_val = self.state[self.current_step - 1]
                mask[0:prev_val+1] = 0 # 이전 숫자와 같거나 작은 것은 마스킹 (조합 최적화)
        else:
            # 4칸을 다 채웠다면, 오직 '제출' 버튼만 활성화
            mask[self.submit_action] = 1
            
        return mask

    def _get_info(self):
        return {
            "action_mask": self.get_valid_action_mask(),
            "current_step": self.current_step
        }
    
    def _get_final_reward(self):
        """제출 시, 완성된 4개의 숫자를 행렬로 바꾸고 채점합니다."""
        if self.evaluator is None: 
            return 0.0
            
        a1, a2, b1, b2 = self.state
        
        # 방금 만든 수학 엔진으로 H_X, H_Z 조립!
        Hx, Hz = self.builder.build_matrices(a1, a2, b1, b2)
        
        # 🌟 다음 단계에서 수정할 qldpc_reward_calculator 호출
        from train.module.reward_calculator import calculate_qldpc_reward
        reward_res = calculate_qldpc_reward(Hx, Hz, self.L, self.evaluator)
        
        return reward_res["final_value"]

    def get_uniform_backward_prob(self):
        """
        [GFlowNet 마법] 역방향 확률 P_B
        현재 상태에 도달하기 위한 직전 상태의 경우의 수.
        우리는 '순서대로' 채워넣었기 때문에, 직전 상태는 무조건 방금 넣은 숫자를 지운 상태 딱 1개입니다.
        따라서 역방향 확률은 항상 1.0 (100%) 입니다!
        """
        return 1.0