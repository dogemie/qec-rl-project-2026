"""_summary_
실행 방법 1 (시드 지정): python train.py --seed 42
실행 방법 2 (시드 자동): python train.py
"""

import os
import time
import logging
import argparse
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# 우리가 만든 커스텀 모듈들
from envs.qec_env import QECEnv
from agent.network import QECNet
from agent.mcts import MCTS
from sim.stim_interface import StimEvaluator
from utils.viz import draw_surface_code_style 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"🌱 [Seed Fix] 모든 난수 시드가 {seed}로 고정되었습니다. (재현성 보장)")


class AlphaZeroTrainer:
    def __init__(self, seed):
        self.seed = seed
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M")
        
        # 🌟 폴더명에 시드값 추가 (예: 260314_1645_s77)
        self.run_name = f"{self.timestamp}_s{self.seed}"
        self.run_dir = os.path.join("outputs", self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        os.makedirs("logging", exist_ok=True)
        log_file = os.path.join("logging", f"train_log_{self.run_name}.txt")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(log_file, encoding='utf-8')]
        )
        self.logger = logging.getLogger(__name__)
        
        self.num_qubits = 7
        self.num_stabilizers = 3
        self.episodes = 200           
        self.epochs = 100              
        self.mcts_simulations = 200   
        self.batch_size = 32          
        
        self.env = QECEnv(num_qubits=self.num_qubits, num_stabilizers=self.num_stabilizers)
        self.evaluator = StimEvaluator(num_qubits=self.num_qubits, noise_rate=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = QECNet(self.num_qubits, self.num_stabilizers).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001, weight_decay=1e-4)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.memory = deque(maxlen=10000) 
        
        self.best_logical_error = 1.0 
        self.best_Hx = None
        self.best_Hz = None

    def _console_print(self, message):
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        print(f"{time_str} [INFO] {message}")

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
        dot_product = np.dot(Hx, Hz.T)
        
        violations = np.sum((dot_product % 2) != 0)
        
        orphaned_x = np.sum(np.sum(Hx, axis=0) == 0)
        orphaned_z = np.sum(np.sum(Hz, axis=0) == 0)
        total_orphans = orphaned_x + orphaned_z
        
        steps = len(episode_memory) 
        
        if violations > 0 or total_orphans > 0:
            max_violations = self.num_stabilizers * self.num_stabilizers
            penalty_score = (violations + total_orphans) / (max_violations + self.num_qubits)
            final_value = -1.0 * penalty_score
            self.logger.info(f"⚠️ [제출 완료] {steps}턴 진행 -> 위반 {violations}개, 버려진 큐비트 {total_orphans}개 (가치: {final_value:.2f})")
        else:
            logical_error = self.evaluator.evaluate_logical_error_rate(Hx, Hz)
            baseline_error = 0.01
            
            if logical_error >= 1.0:
                final_value = 0.0 
            else:
                improvement = (baseline_error - logical_error) / baseline_error
                base_value = np.clip(improvement, 0.1, 1.0) 
                
                # 🌟 수정 1: 인간의 편향인 대칭성 보너스 제거
                # 🌟 수정 2: 보편적 진리인 희소성(Sparsity/LDPC) 보상 적용
                total_elements = Hx.size + Hz.size
                total_ones = np.sum(Hx) + np.sum(Hz)
                sparsity_ratio = 1.0 - (total_ones / total_elements) # 1이 적을수록 1.0에 가까워짐
                
                # 에러율 개선도(80%) + 희소성(20%) 가중합 (비율은 조절 가능합니다)
                final_value = (base_value * 0.8) + (sparsity_ratio * 0.2)
                self.logger.info(f"💎 [희소성 보상] 선 밀도 최소화! Sparsity: {sparsity_ratio:.2f}")
            
            self.logger.info(f"✨ [기적의 코드] {steps}턴 진행! 완벽한 규칙 통과! 논리 에러율: {logical_error:.4f} -> 가치: {final_value:.2f}")
            
            if logical_error < self.best_logical_error and logical_error < baseline_error:
                self.best_logical_error = logical_error
                self.best_Hx = Hx.copy()
                self.best_Hz = Hz.copy()
                
                msg = f"🏆 [신기록 달성] 에러율: {logical_error:.4f} (위치: Epoch {current_epoch+1}, Ep {current_ep+1})"
                self.logger.info(msg)
                self._console_print(msg)
                
                folder_name = f"best_codes_epc{current_epoch+1}_ep{current_ep+1}"
                save_dir = os.path.join(self.run_dir, folder_name)
                os.makedirs(save_dir, exist_ok=True)
                
                np.save(os.path.join(save_dir, "best_Hx.npy"), Hx)
                np.save(os.path.join(save_dir, "best_Hz.npy"), Hz)
                draw_surface_code_style(Hx, Hz, save_dir, filename_prefix="best_tanner_graph")
                self.evaluator.save_circuit_diagram(Hx, Hz, save_dir, filename="best_circuit.svg")
        
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
        
        total_loss = value_loss + policy_loss
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), value_loss.item(), policy_loss.item()

    def run(self):
        start_msg = f"🚀 학습을 시작합니다! 장치: {self.device} (강력한 탐색 모드)"
        self.logger.info(start_msg)
        self._console_print(start_msg)
        
        for epoch in range(self.epochs):
            epoch_msg = f"=== Epoch {epoch+1}/{self.epochs} ==="
            self.logger.info(epoch_msg)
            self._console_print(epoch_msg)
            
            for ep in range(self.episodes):
                episode_data = self.execute_episode(epoch, ep)
                self.memory.extend(episode_data)
                
            losses = None
            for _ in range(20): 
                losses = self.train_network()
            self.lr_scheduler.step()
                
            if losses:
                loss_msg = f"📈 Loss - Total: {losses[0]:.4f} | Value: {losses[1]:.4f} | Policy: {losses[2]:.4f}"
                self.logger.info(loss_msg)
                self._console_print(loss_msg)
                
        model_path = os.path.join(self.run_dir, f"qec_alphazero_model_s{self.seed}.pth")
        torch.save(self.network.state_dict(), model_path)
        
        if self.best_Hx is not None and self.best_Hz is not None:
            final_dir = os.path.join(self.run_dir, "final_codes")
            os.makedirs(final_dir, exist_ok=True)
            
            np.save(os.path.join(final_dir, "final_Hx.npy"), self.best_Hx)
            np.save(os.path.join(final_dir, "final_Hz.npy"), self.best_Hz)
            draw_surface_code_style(self.best_Hx, self.best_Hz, final_dir, filename_prefix="final_tanner_graph")
            
            self.evaluator.evaluate_logical_error_rate(self.best_Hx, self.best_Hz)
            self.evaluator.save_circuit_diagram(self.best_Hx, self.best_Hz, final_dir, filename="final_circuit.svg")
            
            final_msg = f"🌟 [최종 결과] 가장 뛰어났던 코드가 'final_codes' 폴더에 최종 정리되었습니다. (최종 에러율: {self.best_logical_error:.4f})"
            self.logger.info(final_msg)
            self._console_print(final_msg)
        
        end_msg = f"🎉 학습 완료! 최고 에러율: {self.best_logical_error:.4f} \n저장 위치: {model_path}"
        self.logger.info(end_msg)
        self._console_print(end_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero 기반 양자 오류 정정 코드 탐색기")
    
    # 🌟 seed 인자를 선택사항(default=None)으로 변경
    parser.add_argument('--seed', type=int, default=None, 
                        help="실험의 완벽한 재현성을 위한 난수 시드값 (입력하지 않으면 자동 할당)")
    args = parser.parse_args()
    
    # 🌟 outputs 폴더를 뒤져서 이미 사용한 시드 목록 확보
    used_seeds = set()
    if os.path.exists("outputs"):
        for folder_name in os.listdir("outputs"):
            if "_s" in folder_name:
                try:
                    # '260314_1916_s77' 형태에서 마지막 77 추출
                    seed_val = int(folder_name.split("_s")[-1])
                    used_seeds.add(seed_val)
                except ValueError:
                    continue
    
    # 시드가 입력되지 않았다면 1~99999 중 사용하지 않은 시드 자동 뽑기
    if args.seed is None:
        available_seeds = list(set(range(1, 100000)) - used_seeds)
        if not available_seeds:
            final_seed = 42 # 만약 모든 시드가 꽉 찼다면 기본값
        else:
            final_seed = random.choice(available_seeds)
        print(f"🎲 시드가 입력되지 않아, 사용되지 않은 시드 {final_seed}번을 자동 할당합니다.")
    else:
        final_seed = args.seed
    
    set_seed(final_seed)
    
    # Trainer에 시드값을 넘겨줌
    trainer = AlphaZeroTrainer(seed=final_seed)
    trainer.run()