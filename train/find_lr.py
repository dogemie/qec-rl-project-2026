import os
import sys
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 🌟 1. 시스템 경로(sys.path)에 최상위 루트 폴더 강제 추가
# find_lr.py가 train/ 폴더 안에 있으므로, 한 단계 위('..')를 루트로 설정합니다.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 🌟 2. 루트 경로가 추가된 이후에 프로젝트 내부 모듈 임포트
# (위에서 PROJECT_ROOT를 구했으므로 from config import PROJECT_ROOT는 생략합니다)
from utils.helpers import set_seed  # 실제 helpers.py 위치에 맞게 유지
from train.module.trainer import AlphaZeroTrainer

def run_lr_finder(trainer, start_lr=1e-6, end_lr=0.1, num_iters=100):
    print("🚀 [LR Finder] 탐색을 시작합니다.")
    
    target_data_size = trainer.batch_size * 2
    
    # 🌟 1. tqdm 프로그레스 바 적용 및 메모리 저장 버그 수정
    with tqdm(total=target_data_size, desc="⏳ 더미 데이터 수집") as pbar:
        while len(trainer.memory) < target_data_size:
            # 에피소드를 실행하고 반환된 데이터(Trajectory)를 받음
            trajectory = trainer.execute_episode(0, 0)
            
            if trajectory:
                # 🌟 핵심: 받은 데이터를 실제로 메모리에 추가(extend)
                trainer.memory.extend(trajectory)
                
                # 프로그레스 바 게이지 채우기
                current_len = len(trainer.memory)
                update_amount = current_len - pbar.n
                if update_amount > 0:
                    # 목표치를 넘어가서 UI가 깨지는 것을 방지하기 위해 min 사용
                    pbar.update(min(update_amount, target_data_size - pbar.n))
    print("✅ 데이터 수집 완료! 본격적인 탐색을 시작합니다.")

    print("📈 학습률(LR) 지수적 증가 테스트 시작...")
    optimizer = trainer.optimizer
    
    # 옵티마이저 학습률 초기화
    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
        
    lr_multiplier = (end_lr / start_lr) ** (1 / num_iters)
    
    lrs = []
    losses = []
    best_loss = float('inf')
    
    for i in range(num_iters):
        # trainer 내부의 train_network를 1회 호출
        loss_vals = trainer.train_network()
        if loss_vals is None: 
            continue
            
        total_loss = loss_vals[0]
        current_lr = optimizer.param_groups[0]['lr']
        
        lrs.append(current_lr)
        losses.append(total_loss)
        
        print(f"Iter {i+1}/{num_iters} | LR: {current_lr:.6f} | Loss: {total_loss:.4f}")
        
        # Loss 폭발 방지 (기존 베스트보다 5배 이상 튀면 중단)
        if total_loss > best_loss * 5 and i > 10:
            print("⚠️ Loss 폭발 감지! 탐색 조기 종료.")
            break
            
        if total_loss < best_loss:
            best_loss = total_loss
            
        # 학습률 지수적 증가
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_multiplier
            
    # 시각화 및 저장
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, marker='o', markersize=3)
    plt.xscale('log')
    plt.xlabel('Learning Rate (Log Scale)')
    plt.ylabel('Total Loss')
    plt.title('LR Finder (Find the steepest descending gradient)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    save_path = 'lr_finder_result.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 완료! '{save_path}' 그래프를 확인하세요.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="학습률(LR) 최적점 탐색기")
    parser.add_argument('--seed', type=int, default=None, help="시드값")
    args = parser.parse_args()
    
    # --- main.py와 동일한 시드 탐색 및 세팅 로직 ---
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
        final_seed = 42 if not available_seeds else random.choice(available_seeds)
        print(f"🎲 LR Finder 용 시드 {final_seed}번 자동 할당")
    else:
        final_seed = args.seed
    
    set_seed(final_seed)
    
    # 🌟 1. Trainer 초기화 (학습 준비 완료 상태)
    trainer = AlphaZeroTrainer(seed=final_seed)
    
    # 🌟 2. 본 학습(run) 대신 LR Finder 함수 실행!
    run_lr_finder(trainer, start_lr=1e-6, end_lr=0.1, num_iters=100)