import os
import csv
import logging
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm

from utils.viz import draw_surface_code_style 
from utils.viz_fano import draw_fano_steane_graph
from utils.viz_2d_grid import draw_2d_grid_layout
from utils.sinter_runner import run_sinter_evaluation

class ExperimentLogger:
    """
    강화학습 실험 중 발생하는 모든 파일 I/O, 로깅, 시각화 저장을 전담하는 클래스입니다.
    """
    def __init__(self, project_root, seed):
        self.project_root = project_root
        self.seed = seed
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M")
        self.run_name = f"{self.timestamp}_s{self.seed}"
        
        # 1. 출력(outputs) 및 로깅(logging) 폴더 세팅
        self.run_dir = os.path.join(self.project_root, "outputs", self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        log_dir = os.path.join(self.project_root, "logging")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_log_{self.run_name}.txt")
        
        # 2. 로거(Logger) 세팅
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(log_file, encoding='utf-8')]
        )
        self.logger = logging.getLogger(__name__)
        
        # 3. CSV 파일 초기화
        self.csv_file = os.path.join(self.run_dir, f"training_history_{self.run_name}.csv")
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Total_Loss', 'Value_Loss', 'Policy_Loss', 
                             'Best_Error_1%', 'Best_Error_0.1%', 'Best_Error_5%', 
                             'Best_CNOT_Count', 'Best_Wiring_Distance'])

    def info(self, message):
        """콘솔(tqdm)과 로그 파일에 동시에 메시지를 출력합니다."""
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        tqdm.write(f"{time_str} [INFO] {message}")
        self.logger.info(message)

    def log_epoch_to_csv(self, epoch, losses, best_err, best_cnots, best_distance):
        """에포크가 끝날 때마다 CSV에 학습 상태를 한 줄 추가합니다."""
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, losses[0], losses[1], losses[2], 
                             best_err, '-', '-', 
                             best_cnots, best_distance])

    def save_model(self, network):
        """PyTorch 신경망 가중치(.pth)를 저장합니다."""
        model_path = os.path.join(self.run_dir, f"qec_alphazero_model_s{self.seed}.pth")
        torch.save(network.state_dict(), model_path)
        return model_path

    def _generate_and_save_artifacts(self, save_dir, Hx, Hz, evaluator):
        """행렬 저장 및 4종 시각화(Tanner, Circuit, Fano, 2D Grid) 자동 수행"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Numpy 행렬 저장
        hx_save_path = os.path.join(save_dir, "Hx.npy")
        hz_save_path = os.path.join(save_dir, "Hz.npy")
        np.save(hx_save_path, Hx)
        np.save(hz_save_path, Hz)
        
        # 2. Tanner Graph
        draw_surface_code_style(Hx, Hz, save_dir, filename_prefix="tanner_graph")
        
        # 3. Stim Circuit
        evaluator.save_circuit_diagram(Hx, Hz, save_dir, filename="circuit.svg")
        
        # 4. Fano Plane
        fano_save_path = os.path.join(save_dir, "fano_comparison.png")
        draw_fano_steane_graph(hx_save_path, hz_save_path, save_path=fano_save_path, show_plot=False)
        
        # 5. Hardware 2D Grid
        grid_save_path = os.path.join(save_dir, "hardware_2d_grid.png")
        draw_2d_grid_layout(hx_save_path, hz_save_path, save_path=grid_save_path, show_plot=False)

    def save_best_code(self, epoch, ep, Hx, Hz, evaluator):
        """새로운 신기록 달성 시 코드를 저장합니다."""
        folder_name = f"best_codes_epc{epoch}_ep{ep}"
        save_dir = os.path.join(self.run_dir, folder_name)
        self._generate_and_save_artifacts(save_dir, Hx, Hz, evaluator)

    def save_final_codes(self, best_Hx, best_Hz, evaluator):
        """학습이 모두 끝난 후 최종 베스트 코드를 저장합니다."""
        if best_Hx is None or best_Hz is None:
            return
        final_dir = os.path.join(self.run_dir, "final_codes")
        self._generate_and_save_artifacts(final_dir, best_Hx, best_Hz, evaluator)
    
    def trigger_sinter_evaluation(self, best_Hx, best_Hz, evaluator):
        """Break-even 돌파 시 외부 Sinter 모듈을 가동합니다."""
        self.info("🚀 [Break-Even 달성] 물리적 에러율(1%)의 한계를 돌파한 기적의 코드가 발견되었습니다!")
        run_sinter_evaluation(best_Hx, best_Hz, evaluator, save_dir=self.run_dir, logger=self)