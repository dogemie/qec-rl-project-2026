"""_summary_
실행 방법 1 (시드 지정): python train.py --seed 42
실행 방법 2 (시드 자동): python train.py
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

# 분리해 둔 모듈에서 클래스 임포트
from train.module.trainer import AlphaZeroTrainer
from utils.helpers import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero 기반 양자 오류 정정 코드 탐색기")
    parser.add_argument('--seed', type=int, default=None, help="실험의 완벽한 재현성을 위한 난수 시드값")
    args = parser.parse_args()
    
    # 🌟 시드 중복 검사를 위해 PROJECT_ROOT 기준 outputs 폴더 뒤지기
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
    
    trainer = AlphaZeroTrainer(seed=final_seed)
    trainer.run()