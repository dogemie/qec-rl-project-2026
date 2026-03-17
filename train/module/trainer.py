import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from tqdm import tqdm

# 🌟 1. 프로젝트 최상위 루트 경로를 절대 경로로 계산하여 파이썬 시스템 경로에 추가
# 이 파일이 `train/trainer.py`에 위치하므로, 한 단계 위('..')가 프로젝트 루트가 됩니다.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 기존 강화학습 및 시뮬레이션 환경 임포트
from envs.qec_env import QECEnv
from agent.network import QECNet
from agent.mcts import MCTS
from sim.stim_interface import StimEvaluator

# 🌟 2. 새롭게 분리한 깔끔한 모듈들 임포트!
from train.module.reward_calculator import calculate_qec_reward
from utils.experiment_logger import ExperimentLogger

class AlphaZeroTrainer:
    def __init__(self, seed):
        self.seed = seed
        
        # 🌟 3. 모든 파일 저장, 로깅, 시각화 출력을 전담하는 로거 객체 생성
        self.logger = ExperimentLogger(PROJECT_ROOT, seed)
        
        self.best_cnots = -1
        self.best_distance = -1
        
        # Surface Code [9,1,3] 테스트 환경
        self.num_qubits = 9
        self.num_stabilizers = 4
        self.episodes = 200           
        self.epochs = 100              
        self.mcts_simulations = 200   
        self.batch_size = 32          
        
        self.env = QECEnv(num_qubits=self.num_qubits, num_stabilizers=self.num_stabilizers)
        self.evaluator = StimEvaluator(num_qubits=self.num_qubits, noise_rate=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = QECNet(self.num_qubits, self.num_stabilizers).to(self.device)
        
        # 동적 가중치(Uncertainty Weighting) 파라미터
        self.log_var_v = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_var_p = torch.zeros(1, requires_grad=True, device=self.device)

        self.optimizer = optim.AdamW(
            list(self.network.parameters()) + [self.log_var_v, self.log_var_p], 
            lr=0.01, 
            weight_decay=1e-4
        )
        
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=25, T_mult=1, eta_min=1e-4
        )
        
        self.memory = deque(maxlen=10000)
        
        self.best_logical_error = 1.0 
        self.best_Hx = None
        self.best_Hz = None

    def execute_episode(self, current_epoch, current_ep):
        state, info = self.env.reset()
        episode_memory = []
        
        while True:
            if np.sum(info['action_mask']) == 0:
                break
            
            mcts = MCTS(self.network, self.env, num_simulations=self.mcts_simulations)
            action_probs = mcts.search(state)
            
            episode_memory.append([state.copy(), action_probs, info['action_mask']])
            
            action = np.random.choice(len(action_probs), p=action_probs)
            state, step_reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
                
        Hx, Hz = state[0], state[1]
        steps = len(episode_memory) 
        
        # 🌟 4. 분리된 보상 계산기 호출! (복잡한 100줄의 로직이 단 한 줄로 압축)
        reward_result = calculate_qec_reward(Hx, Hz, self.num_qubits, self.num_stabilizers, self.evaluator)
        
        final_value = reward_result["final_value"]
        
        # 유효한 코드가 나왔을 때만 로그 출력 및 신기록 검사
        if reward_result["is_valid"]:
            err_01 = reward_result["err_01"]
            total_cnots = reward_result["cnots"]
            total_distance = reward_result["distance"]
            
            self.logger.info(f"💎 [HW최적화] CNOT: {total_cnots} | 배선 거리: {total_distance} | Willow급 에러율: {reward_result['err_001']:.5f}")
            self.logger.info(f"✨ [기적의 코드] {steps}턴 진행! 종합 가치: {final_value:.2f} (에러율 1%기준: {err_01:.4f})")
            
            # 신기록 저장 로직
            if err_01 < self.best_logical_error and err_01 < 0.01:
                self.best_logical_error = err_01
                self.best_Hx = Hx.copy()
                self.best_Hz = Hz.copy()
                self.best_cnots = total_cnots
                self.best_distance = total_distance
                
                self.logger.info(f"🏆 [신기록 달성] 에러율(1%기준): {err_01:.4f} (위치: Epoch {current_epoch+1}, Ep {current_ep+1})")
                
                # 시각화 및 numpy 행렬 자동 저장 (로거에 위임)
                self.logger.save_best_code(current_epoch + 1, current_ep + 1, Hx, Hz, self.evaluator)
        
        # 메모리에 이번 에피소드의 최종 가치(Value) 저장
        for step_data in episode_memory:
            step_data.append(final_value)
            
        return episode_memory

    def train_network(self):
        if len(self.memory) < self.batch_size: return None
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([data[0] for data in mini_batch])).to(self.device)
        target_probs = torch.FloatTensor(np.array([data[1] for data in mini_batch])).to(self.device)
        masks = torch.FloatTensor(np.array([data[2] for data in mini_batch])).to(self.device)
        target_values = torch.FloatTensor(np.array([data[3] for data in mini_batch])).unsqueeze(1).to(self.device)
        
        self.optimizer.zero_grad()
        pred_probs, pred_values = self.network(states, masks)
        
        value_loss = F.mse_loss(pred_values, target_values)
        policy_loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-8)) / self.batch_size
        
        log_var_v_clamped = torch.clamp(self.log_var_v, min=-3.0, max=3.0)
        log_var_p_clamped = torch.clamp(self.log_var_p, min=-3.0, max=3.0)
        
        weighted_value_loss = torch.exp(-log_var_v_clamped) * value_loss + log_var_v_clamped
        weighted_policy_loss = torch.exp(-log_var_p_clamped) * policy_loss + log_var_p_clamped
        
        total_loss = weighted_value_loss + weighted_policy_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), value_loss.item(), policy_loss.item()

    def run(self):
        self.logger.info(f"🚀 학습을 시작합니다! 장치: {self.device} (강력한 탐색 모드)")
        
        for epoch in range(self.epochs):
            self.logger.info(f"=== Epoch {epoch+1}/{self.epochs} ===")
            
            for ep in tqdm(range(self.episodes), desc="🎮 데이터 수집 (Self-Play)", leave=False, ncols=90, colour='green'):
                episode_data = self.execute_episode(epoch, ep)
                self.memory.extend(episode_data)
                
            losses = None
            for _ in tqdm(range(100), desc="🧠 신경망 학습 (Training)", leave=False, ncols=90, colour='blue'): 
                losses = self.train_network()
                
            self.lr_scheduler.step()
                
            if losses:
                current_lr = self.optimizer.param_groups[0]['lr']
                weight_v = torch.exp(-self.log_var_v).item()
                weight_p = torch.exp(-self.log_var_p).item()
                
                loss_msg = (f"📈 Loss - Tot: {losses[0]:.4f} | Val: {losses[1]:.4f} | Pol: {losses[2]:.4f} | "
                            f"LR: {current_lr:.6f} | W_Val: {weight_v:.3f}, W_Pol: {weight_p:.3f}")
                self.logger.info(loss_msg)
                
                # CSV 기록 (로거에 위임)
                self.logger.log_epoch_to_csv(epoch + 1, losses, self.best_logical_error, self.best_cnots, self.best_distance)
            
        # 🌟 5. 학습이 모두 끝난 후 저장 (로거에 위임)
        model_path = self.logger.save_model(self.network)
        self.logger.save_final_codes(self.best_Hx, self.best_Hz, self.evaluator)
            
        self.logger.info(f"🌟 [최종 결과] 가장 뛰어났던 코드가 'final_codes' 폴더에 정리되었습니다. (최종 에러율: {self.best_logical_error:.4f})")
        
        # 🌟 6. Break-even 달성 시 Sinter 정밀 검증 가동!
        if self.best_logical_error < 0.01:
            self.logger.info("🚀 [Break-Even 달성] 물리적 에러율(1%)의 한계를 돌파한 기적의 코드가 발견되었습니다!")
            self.logger.info("🔥 Sinter를 사용한 Threshold 정밀 검증 및 그래프 생성을 시작합니다. (시간이 조금 걸릴 수 있습니다...)")
            
            # 로거의 Sinter 평가 함수 호출
            self.logger.trigger_sinter_evaluation(self.best_Hx, self.best_Hz, self.evaluator)
        
        self.logger.info(f"🎉 학습 완료! 최고 에러율: {self.best_logical_error:.4f} \n저장 위치: {model_path}")