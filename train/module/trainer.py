import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from envs.qec_env import QECEnv
# 🌟 [수정] 기존 QECNet과 MCTS를 버리고, GFlowNet 모델을 불러옵니다.
from agent.gflownet import QEC_GFlowNet
from sim.stim_interface import StimEvaluator
from train.module.reward_calculator import calculate_qec_reward
from utils.experiment_logger import ExperimentLogger

class GFlowNetTrainer:
    def __init__(self, seed):
        self.seed = seed
        self.logger = ExperimentLogger(PROJECT_ROOT, seed)
        
        self.best_cnots = -1
        self.best_distance = -1
        
        self.num_qubits = 9
        self.num_stabilizers = 4
        self.episodes = 200           
        self.epochs = 100              
        self.batch_size = 32          
        
        self.evaluator = StimEvaluator(num_qubits=self.num_qubits, noise_rate_x=0.01, noise_rate_z=0.01)
        self.env = QECEnv(num_qubits=self.num_qubits, num_stabilizers=self.num_stabilizers, evaluator=self.evaluator)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 🌟 [수정] 신경망 교체
        self.network = QEC_GFlowNet(self.num_qubits, self.num_stabilizers).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.network.parameters(), 
            lr=0.005, # GFlowNet은 학습률을 약간 낮추는 것이 안정적입니다.
            weight_decay=1e-4
        )
        
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=25, T_mult=1, eta_min=1e-4
        )
        
        # 🌟 [수정] 메모리에는 이제 턴(Turn) 단위가 아니라 '하나의 완벽한 궤적(Trajectory)' 전체를 담습니다.
        self.memory = deque(maxlen=2000) 
        
        self.best_logical_error = 1.0 
        self.best_Hx = None
        self.best_Hz = None

    def execute_episode(self, current_epoch, current_ep):
        state, info = self.env.reset()
        trajectory = []
        
        # 🌟 MCTS 없이 초고속으로 전진(Forward Sampling)합니다!
        while True:
            mask = info['action_mask']
            
            # 1. 신경망에 상태를 넣고 확률 분포(P_F)를 얻습니다. (기울기 계산 금지)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                logits = self.network(state_tensor).squeeze(0)
                
                # 마스킹된 불가능한 행동은 확률을 0(-inf)으로 만듭니다.
                mask_tensor = torch.FloatTensor(mask).to(self.device)
                logits = logits.masked_fill(mask_tensor == 0, -1e9)
                probs = F.softmax(logits, dim=0)
            
            # 2. 확률 분포에 따라 행동을 랜덤 샘플링합니다! (GFlowNet의 핵심: 탐색의 다양성)
            action = torch.multinomial(probs, 1).item()
            
            # 3. 환경 진행 및 역방향 확률(P_B) 획득
            next_state, reward, terminated, truncated, next_info = self.env.step(action)
            P_B = self.env.get_uniform_backward_prob()
            
            # 4. 궤적 기록 (Loss 계산 시 기울기를 다시 구하기 위해 상태와 액션만 저장)
            trajectory.append({
                'state': state.copy(),
                'action': action,
                'mask': mask.copy(),
                'log_PB': np.log(P_B + 1e-8)
            })
            
            state = next_state
            info = next_info
            
            if terminated or truncated:
                break
                
        # 🌟 에피소드 종료 후, 우리가 개조한 reward_calculator로 0보다 큰 양수 보상(R) 획득
        Hx, Hz = state[0], state[1]
        reward_result = calculate_qec_reward(Hx, Hz, self.num_qubits, self.num_stabilizers, self.evaluator)
        final_reward_R = reward_result["final_value"] # 지수 함수가 적용된 양수값!
        
        # 로깅 및 신기록 저장 로직 (기존과 동일)
        if reward_result["is_valid"]:
            err_01 = reward_result["err_01"]
            total_cnots = reward_result["cnots"]
            
            if err_01 < self.best_logical_error and err_01 < 0.1:
                self.best_logical_error = err_01
                self.best_Hx = Hx.copy()
                self.best_Hz = Hz.copy()
                self.logger.info(f"🏆 [신기록 달성] 에러율: {err_01:.4f} (에포크 {current_epoch+1})")
                self.logger.save_best_code(current_epoch + 1, current_ep + 1, Hx, Hz, self.evaluator)
        
        # 메모리에 '완성된 궤적 1개'와 '최종 보상 R'을 통째로 저장
        self.memory.append((trajectory, final_reward_R))
        
        # 🌟 [수정] 단순 길이 반환이 아니라, 로깅을 위한 상세 성적표 반환!
        return {
            "reward": final_reward_R,
            "violations": reward_result["violations"],
            "orphans": reward_result["orphans"],
            "steps": len(trajectory),
            "is_valid": reward_result["is_valid"]
        }
        
    def train_network(self):
        if len(self.memory) < self.batch_size: return None
        mini_batch = random.sample(self.memory, self.batch_size)
        
        self.optimizer.zero_grad()
        
        # 🌟 GFlowNet의 특별한 파라미터 (전체 분배 함수)
        logZ = self.network.logZ 
        
        tb_losses = []
        for trajectory, R in mini_batch:
            sum_log_PF = 0.0
            sum_log_PB = 0.0
            
            # 궤적을 처음부터 끝까지 다시 따라가며 현재 신경망의 P_F 확률(기울기 포함)을 누적합니다.
            for step in trajectory:
                state_t = torch.FloatTensor(step['state']).unsqueeze(0).to(self.device)
                mask_t = torch.FloatTensor(step['mask']).to(self.device)
                
                logits = self.network(state_t).squeeze(0)
                logits = logits.masked_fill(mask_t == 0, -1e9)
                log_probs = F.log_softmax(logits, dim=0)
                
                sum_log_PF = sum_log_PF + log_probs[step['action']]
                sum_log_PB = sum_log_PB + step['log_PB']
                
            # 최종 보상 R의 로그값
            log_R = torch.log(torch.tensor(R, dtype=torch.float32).to(self.device) + 1e-8)
            
            # 🌟 [Trajectory Balance Loss 핵심 수식]
            # (log Z + 총 Forward 확률 - 총 Backward 확률 - log R)^2
            loss_i = (logZ + sum_log_PF - sum_log_PB - log_R) ** 2
            tb_losses.append(loss_i)
            
        # 배치 평균 Loss 계산 및 역전파
        total_loss = torch.stack(tb_losses).mean()
        total_loss.backward()
        self.optimizer.step()
        
        # 기존 로깅 포맷을 맞추기 위해 3개 리턴 (Val, Pol은 더 이상 안 쓰므로 0.0 처리)
        return total_loss.item(), 0.0, 0.0 

    def run(self):
        self.logger.info(f"🚀 GFlowNet 학습을 시작합니다! 장치: {self.device} (생성형 탐색 모드)")
        
        for epoch in range(self.epochs):
            self.logger.info(f"=== Epoch {epoch+1}/{self.epochs} ===")
            
            # 🌟 [수정] 200판 동안의 성적을 기록할 장부 준비
            stats = {"reward": [], "violations": [], "orphans": [], "steps": [], "valid_count": 0}
            
            # 데이터 수집 
            for ep in tqdm(range(self.episodes), desc="🎮 궤적 샘플링", leave=False, ncols=90, colour='green'):
                ep_result = self.execute_episode(epoch, ep)
                
                # 장부에 기록
                stats["reward"].append(ep_result["reward"])
                stats["violations"].append(ep_result["violations"])
                stats["orphans"].append(ep_result["orphans"])
                stats["steps"].append(ep_result["steps"])
                if ep_result["is_valid"]:
                    stats["valid_count"] += 1
                    
            # 🌟 [수정] 200판 수집 완료 후, 에포크 요약 정보(계기판) 출력!
            avg_v = np.mean(stats["violations"])
            avg_o = np.mean(stats["orphans"])
            avg_s = np.mean(stats["steps"])
            max_r = np.max(stats["reward"])
            
            self.logger.info(f"📊 [상태] 평균 위반: {avg_v:.1f}개 | 고아: {avg_o:.1f}개 | 턴: {avg_s:.1f} | 최고보상: {max_r:.5f} | 정답: {stats['valid_count']}개")
                
            # 신경망 학습
            losses = None
            for _ in tqdm(range(100), desc="🧠 TB Loss 최적화", leave=False, ncols=90, colour='blue'): 
                losses = self.train_network()
                
            self.lr_scheduler.step()
                
            if losses:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"📈 Trajectory Balance Loss: {losses[0]:.4f} | logZ: {self.network.logZ.item():.4f} | LR: {current_lr:.6f}")
                
        # 최종 저장 로직 (기존과 동일)
        model_path = self.logger.save_model(self.network)
        self.logger.save_final_codes(self.best_Hx, self.best_Hz, self.evaluator)
        self.logger.info(f"🎉 GFlowNet 학습 완료! 최고 에러율: {self.best_logical_error:.4f}")