import os
import sys
import time
import logging
import csv
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# 🌟 1. 프로젝트 최상위 루트 경로를 절대 경로로 계산하여 파이썬 시스템 경로에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 이제 루트 경로에 있는 커스텀 모듈들을 정상적으로 임포트할 수 있습니다.
from envs.qec_env import QECEnv
from agent.network import QECNet
from agent.mcts import MCTS
from sim.stim_interface import StimEvaluator
from utils.viz import draw_surface_code_style 
from utils.viz_fano import draw_fano_steane_graph
from utils.viz_2d_grid import draw_2d_grid_layout

class AlphaZeroTrainer:
    def __init__(self, seed):
        self.seed = seed
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M")
        
        self.run_name = f"{self.timestamp}_s{self.seed}"
        
        # 🌟 2. 저장 폴더 경로도 PROJECT_ROOT를 기준으로 잡아줍니다.
        self.run_dir = os.path.join(PROJECT_ROOT, "outputs", self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        log_dir = os.path.join(PROJECT_ROOT, "logging")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_log_{self.run_name}.txt")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(log_file, encoding='utf-8')]
        )
        self.logger = logging.getLogger(__name__)
        
        # 🌟 3. CSV 및 데이터 관리 설정
        self.csv_file = os.path.join(self.run_dir, f"training_history_{self.run_name}.csv")
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            # 논문용 Figure를 그리기 위한 헤더(Header) 작성
            writer.writerow(['Epoch', 'Total_Loss', 'Value_Loss', 'Policy_Loss', 
                             'Best_Error_1%', 'Best_Error_0.1%', 'Best_Error_5%', 
                             'Best_CNOT_Count', 'Best_Wiring_Distance'])
        
        # 베스트 코드의 하드웨어 스펙을 기억할 변수 추가
        self.best_cnots = -1
        self.best_distance = -1
        
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
        
        # 🌟 1. 동적 가중치(Uncertainty Weighting)를 위한 학습 가능한 파라미터 생성
        # 초기값은 0으로 시작 (즉, 가중치 exp(0) = 1.0 에서 시작)
        self.log_var_v = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_var_p = torch.zeros(1, requires_grad=True, device=self.device)

        # 🌟 2. Optimizer에 신경망 파라미터와 동적 가중치 파라미터를 함께 등록 (AdamW 사용)
        self.optimizer = optim.AdamW(
            list(self.network.parameters()) + [self.log_var_v, self.log_var_p], 
            lr=0.001, 
            weight_decay=1e-4
        )
        
        # 🌟 3. Cosine Annealing 스케줄러 도입
        # T_max: 반주기(최소점까지 도달하는 에포크 수, 전체 에포크로 설정)
        # eta_min: 가장 작아졌을 때의 학습률 (0.00001로 설정하여 미세조정)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-5
        )
        
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
            # 🌟 1. 다중 노이즈 강건성 평가 (Sycamore + Willow + Worst Case)
            err_01 = self.evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate=0.01)   # 베이스라인 (1%)
            err_001 = self.evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate=0.001) # Willow급 최신 칩 (0.1%)
            err_05 = self.evaluator.evaluate_logical_error_rate(Hx, Hz, noise_rate=0.05)   # 가혹한 환경 (5%)
            
            if err_01 >= 1.0 or err_001 >= 1.0 or err_05 >= 1.0:
                final_value = 0.0
            else:
                # 3가지 환경의 개선도를 종합하여 기본 성능 점수 산출
                imp_01 = max(0, (0.01 - err_01) / 0.01)
                imp_001 = max(0, (0.001 - err_001) / 0.001)
                imp_05 = max(0, (0.05 - err_05) / 0.05)
                base_value = np.clip((imp_01 * 0.5) + (imp_001 * 0.3) + (imp_05 * 0.2), 0.1, 1.0)
                
                # 🌟 2. 2D 기하학적 거리 패널티 (Auto-Grid 알고리즘)
                # 큐비트+안정자 총 노드 수에 맞춰 가상의 정사각형 칩 크기 자동 계산 (예: 13개면 4x4 그리드)
                grid_size = int(np.ceil(np.sqrt(self.num_qubits + self.num_stabilizers * 2)))
                
                def get_coord(index): # 1차원 인덱스를 2D (x, y) 좌표로 변환
                    return (index // grid_size, index % grid_size)
                
                total_distance = 0
                for i in range(Hx.shape[0]):
                    stab_coord = get_coord(self.num_qubits + i) # X 안정자의 가상 좌표
                    for j in range(Hx.shape[1]):
                        if Hx[i, j] == 1:
                            qubit_coord = get_coord(j)
                            # 맨해튼 거리 누적
                            total_distance += abs(stab_coord[0] - qubit_coord[0]) + abs(stab_coord[1] - qubit_coord[1])
                
                for i in range(Hz.shape[0]):
                    stab_coord = get_coord(self.num_qubits + self.num_stabilizers + i) # Z 안정자의 가상 좌표
                    for j in range(Hz.shape[1]):
                        if Hz[i, j] == 1:
                            qubit_coord = get_coord(j)
                            total_distance += abs(stab_coord[0] - qubit_coord[0]) + abs(stab_coord[1] - qubit_coord[1])
                
                # 🌟 3. CNOT 게이트 최소화 (Sparsity)
                total_cnots = np.sum(Hx) + np.sum(Hz)
                
                # 패널티를 0.0 ~ 1.0 사이로 정규화 (선이 짧을수록, 적을수록 1에 가까움)
                distance_penalty = min(1.0, total_distance / (total_cnots * grid_size))
                sparsity_penalty = total_cnots / (self.num_qubits * self.num_stabilizers * 2)
                
                # 최종 가치 점수 = 논리 방어력(60%) + 짧은 배선 보상(20%) + 적은 선 보상(20%)
                final_value = base_value * 0.6 + (1.0 - distance_penalty) * 0.2 + (1.0 - sparsity_penalty) * 0.2
                final_value = np.clip(final_value, 0.0, 1.0)
                
                self.logger.info(f"💎 [HW최적화] CNOT: {total_cnots} | 배선 거리: {total_distance} | Willow급 에러율: {err_001:.5f}")
            
            self.logger.info(f"✨ [기적의 코드] {steps}턴 진행! 종합 가치: {final_value:.2f} (에러율 1%기준: {err_01:.4f})")
            
            # 신기록 저장 판단은 기준치(1%) 에러율을 바탕으로 수행합니다.
            if err_01 < self.best_logical_error and err_01 < 0.01:
                self.best_logical_error = err_01
                self.best_Hx = Hx.copy()
                self.best_Hz = Hz.copy()
                self.best_cnots = total_cnots
                self.best_distance = total_distance
                
                msg = f"🏆 [신기록 달성] 에러율(1%기준): {err_01:.4f} (위치: Epoch {current_epoch+1}, Ep {current_ep+1})"
                self.logger.info(msg)
                self._console_print(msg)
                
                folder_name = f"best_codes_epc{current_epoch+1}_ep{current_ep+1}"
                save_dir = os.path.join(self.run_dir, folder_name)
                os.makedirs(save_dir, exist_ok=True)
                
                hx_save_path = os.path.join(save_dir, "best_Hx.npy")
                hz_save_path = os.path.join(save_dir, "best_Hz.npy")
                np.save(hx_save_path, Hx)
                np.save(hz_save_path, Hz)
                
                draw_surface_code_style(Hx, Hz, save_dir, filename_prefix="best_tanner_graph")
                self.evaluator.save_circuit_diagram(Hx, Hz, save_dir, filename="best_circuit.svg")
                
                # 🌟 Fano 평면 기반의 비교 이미지 자동 생성 및 저장!
                fano_save_path = os.path.join(save_dir, "fano_comparison.png")
                draw_fano_steane_graph(hx_save_path, hz_save_path, save_path=fano_save_path, show_plot=False)
                
                # 🌟 하드웨어 친화적 2D Grid 레이아웃 자동 생성 및 저장!
                grid_save_path = os.path.join(save_dir, "hardware_2d_grid.png")
                draw_2d_grid_layout(hx_save_path, hz_save_path, save_path=grid_save_path, show_plot=False)
                
                # np.save(os.path.join(save_dir, "best_Hx.npy"), Hx)
                # np.save(os.path.join(save_dir, "best_Hz.npy"), Hz)
                # draw_surface_code_style(Hx, Hz, save_dir, filename_prefix="best_tanner_graph")
                # self.evaluator.save_circuit_diagram(Hx, Hz, save_dir, filename="best_circuit.svg")
        
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
        
        # 1. 원본 Loss 계산
        value_loss = F.mse_loss(pred_values, target_values)
        policy_loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-8)) / self.batch_size
        
        # 🌟 2. 수학적 동적 가중치 적용 (Uncertainty Weighting)
        # s 값이 커지면(불확실성이 높으면) 가중치 exp(-s)가 작아져 해당 Loss의 영향을 줄임
        # 반대로 뒤에 + s 가 붙어있어 무한정 s가 커지는 것을 수학적 정규화로 방지
        weighted_value_loss = torch.exp(-self.log_var_v) * value_loss + self.log_var_v
        weighted_policy_loss = torch.exp(-self.log_var_p) * policy_loss + self.log_var_p
        
        # 최종 Loss 합산
        total_loss = weighted_value_loss + weighted_policy_loss
        
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
            for _ in range(100): 
                losses = self.train_network()
                
            # 🌟 에포크가 끝날 때마다 코사인 스케줄러 1스텝 진행 (학습률 부드럽게 감소)
            self.lr_scheduler.step()
                
            if losses:
                # 현재 학습률 확인
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 신경망이 스스로 부여한 현재 가중치(Weight) 확인
                weight_v = torch.exp(-self.log_var_v).item()
                weight_p = torch.exp(-self.log_var_p).item()
                
                # 로그 메시지에 학습률과 동적 가중치 상황을 함께 출력
                loss_msg = (f"📈 Loss - Tot: {losses[0]:.4f} | Val: {losses[1]:.4f} | Pol: {losses[2]:.4f} | "
                            f"LR: {current_lr:.6f} | W_Val: {weight_v:.3f}, W_Pol: {weight_p:.3f}")
                self.logger.info(loss_msg)
                self._console_print(loss_msg)
                
                # 🌟 Episode 마다 CSV에 학습도 기록.
                with open(self.csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, losses[0], losses[1], losses[2], 
                                     self.best_logical_error, '-', '-', 
                                     self.best_cnots, self.best_distance])  
            
            
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
            
            final_msg = f"🌟 [최종 결과] 가장 뛰어났던 코드가 'final_codes' 폴더에 정리되었습니다. (최종 에러율: {self.best_logical_error:.4f})"
            self.logger.info(final_msg)
            self._console_print(final_msg)
        
        end_msg = f"🎉 학습 완료! 최고 에러율: {self.best_logical_error:.4f} \n저장 위치: {model_path}"
        self.logger.info(end_msg)
        self._console_print(end_msg)