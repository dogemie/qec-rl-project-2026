import os
import time
import logging
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# env -> envs 로 수정
from envs.qec_env import QECEnv
from agent.network import QECNet
from agent.mcts import MCTS
from sim.stim_interface import StimEvaluator

class AlphaZeroTrainer:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M")
        
        os.makedirs("logging", exist_ok=True)
        log_file = os.path.join("logging", f"train_log_{self.timestamp}.txt")
        
        # 파일에만 모든 로그를 저장하도록 StreamHandler 제거
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.num_qubits = 7
        self.num_stabilizers = 3
        self.episodes = 50           
        self.epochs = 10             
        self.mcts_simulations = 50   
        self.batch_size = 16
        
        self.env = QECEnv(num_qubits=self.num_qubits, num_stabilizers=self.num_stabilizers)
        self.evaluator = StimEvaluator(num_qubits=self.num_qubits, noise_rate=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = QECNet(self.num_qubits, self.num_stabilizers).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001, weight_decay=1e-4)
        
        self.memory = deque(maxlen=2000)

    def _console_print(self, message):
        """터미널 화면 출력을 위한 깔끔한 포맷팅 헬퍼 함수"""
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        print(f"{time_str} [INFO] {message}")

    def execute_episode(self):
        state, info = self.env.reset()
        episode_memory = []
        
        while True:
            # 🌟 1. 종료 조건 확인: 막다른 길(더 이상 둘 곳이 없음)인가?
            if np.sum(info['action_mask']) == 0:
                # 패널티를 주지 않고 "코드 제출(Submit)"을 위해 루프를 빠져나갑니다!
                break
            
            # 2. 정상적인 MCTS 탐색 및 행동
            mcts = MCTS(self.network, self.env, num_simulations=self.mcts_simulations)
            action_probs = mcts.search(state)
            
            episode_memory.append([state.copy(), action_probs, info['action_mask']])
            
            action = np.random.choice(len(action_probs), p=action_probs)
            state, step_reward, terminated, truncated, info = self.env.step(action)
            
            # 3. 환경 자체 규칙에 의해 게임이 끝난 경우
            if terminated or truncated:
                break
                
        # --- 🌟 4. 최종 심판 (평가 로직) ---
        # 어떻게든 루프를 빠져나왔다면, 지금까지 만든 state를 평가합니다.
        Hx, Hz = state[0], state[1]
        dot_product = np.dot(Hx, Hz.T)
        
        violations = np.sum((dot_product % 2) != 0)
        
        if violations > 0:
            max_violations = self.num_stabilizers * self.num_stabilizers
            final_value = -1.0 * (violations / max_violations)
            
            # 얼마나 잘 버텼는지(턴 수)도 함께 로그에 남겨줍니다.
            steps = len(episode_memory)
            self.logger.info(f"⚠️ [제출 완료] {steps}턴 진행 -> 교환 법칙 위반 {violations}개 (가치: {final_value:.2f})")
        else:
            logical_error = self.evaluator.evaluate_logical_error_rate(Hx, Hz)
            baseline_error = 0.01
            
            if logical_error >= 1.0:
                final_value = 0.0 
            else:
                improvement = (baseline_error - logical_error) / baseline_error
                final_value = np.clip(improvement, 0.1, 1.0) 
            
            self.logger.info(f"✨ [기적의 코드] 완벽한 규칙 통과! 논리 에러율: {logical_error:.4f} -> 가치: {final_value:.2f}")
            
            if logical_error < self.best_logical_error and logical_error < baseline_error:
                self.best_logical_error = logical_error
                msg = f"🏆 [신기록 달성] 새로운 최고 성능 발견! 에러율: {logical_error:.4f}"
                self.logger.info(msg)
                self._console_print(msg)
                
                save_dir = os.path.join(self.run_dir, "best_codes")
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, "best_Hx.npy"), Hx)
                np.save(os.path.join(save_dir, "best_Hz.npy"), Hz)
                draw_surface_code_style(Hx, Hz, save_dir, filename_prefix="best_tanner_graph")
                self.evaluator.save_circuit_diagram(Hx, Hz, save_dir, filename="best_circuit.svg")
        
        # 평가된 최종 점수를 에피소드 메모리의 모든 턴에 반영
        for step_data in episode_memory:
            step_data.append(final_value)
            
        return episode_memory

    def train_network(self):
        if len(self.memory) < self.batch_size:
            return None
            
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
        os.makedirs("outputs", exist_ok=True)
        
        # 파일과 터미널 동시에 기록 (핵심 정보만)
        start_msg = f"🚀 학습을 시작합니다! 장치: {self.device}"
        self.logger.info(start_msg)
        self._console_print(start_msg)
        
        for epoch in range(self.epochs):
            epoch_msg = f"=== Epoch {epoch+1}/{self.epochs} ==="
            self.logger.info(epoch_msg)
            self._console_print(epoch_msg)
            
            for ep in range(self.episodes):
                episode_data = self.execute_episode()
                self.memory.extend(episode_data)
                
            losses = None
            for _ in range(10): 
                losses = self.train_network()
                
            if losses:
                loss_msg = f"📈 Loss - Total: {losses[0]:.4f} | Value: {losses[1]:.4f} | Policy: {losses[2]:.4f}"
                self.logger.info(loss_msg)
                self._console_print(loss_msg)
                
        model_filename = f"qec_alphazero_model_{self.timestamp}.pth"
        model_path = os.path.join("outputs", model_filename)
        torch.save(self.network.state_dict(), model_path)
        
        end_msg = f"🎉 학습 완료! 모델이 '{model_path}'에 저장되었습니다."
        self.logger.info(end_msg)
        self._console_print(end_msg)

if __name__ == "__main__":
    trainer = AlphaZeroTrainer()
    trainer.run()