"""_summary_
실행 방법 1 (시드 지정): python train/main.py --seed 42
실행 방법 2 (시드 자동): python train/main.py
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

# 🌟 프로젝트 루트 경로 인식
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 🌟 AlphaZeroTrainer 대신 새롭게 만든 GFlowNetTrainer를 임포트합니다.
from train.module.trainer import GFlowNetTrainer
from utils.helpers import set_seed

if __name__ == "__main__":
    # 🌟 설명 문구를 GFlowNet 기반으로 변경
    parser = argparse.ArgumentParser(description="GFlowNet 기반 양자 오류 정정(QEC) 생성형 탐색 프레임워크")
    parser.add_argument('--seed', type=int, default=None, help="실험의 완벽한 재현성을 위한 난수 시드값")
    args = parser.parse_args()
    
    # 시드 중복 검사를 위해 PROJECT_ROOT 기준 outputs 폴더 뒤지기
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    used_seeds = set()
    
    if os.path.exists(outputs_dir):
        for folder_name in os.listdir(outputs_dir):
            if "_s" in folder_name:
                try:
                    seed_val = int(folder_name.split("_s")[-1])
                    used_seeds.add(seed_val)
                except ValueError:
                    continue
    
    if args.seed is None:
        available_seeds = list(set(range(1, 100000)) - used_seeds)
        if not available_seeds:
            final_seed = 42 
        else:
            final_seed = random.choice(available_seeds)
        print(f"🎲 시드가 입력되지 않아, 사용되지 않은 시드 {final_seed}번을 자동 할당합니다.")
    else:
        final_seed = args.seed
    
    set_seed(final_seed)
    
    # 🌟 GFlowNetTrainer 객체 생성 및 실행
    trainer = GFlowNetTrainer(seed=final_seed)
    trainer.run()