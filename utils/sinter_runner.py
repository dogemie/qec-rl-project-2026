import os
import csv
import sinter
import multiprocessing
import matplotlib.pyplot as plt

def run_sinter_evaluation(Hx, Hz, evaluator, save_dir, logger=None):
    """
    물리적 에러율(0.1% ~ 5%) 구간에서 코드를 Sinter로 정밀 평가하고,
    Threshold 데이터(CSV) 및 그래프(Plot)를 자동 생성합니다.
    """
    if logger:
        logger.info("🔥 [Sinter Runner] Threshold 정밀 검증(Break-even Test)을 시작합니다...")

    # 1. 평가할 물리적 에러율 구간 설정
    noise_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
    tasks = []
    
    for p in noise_rates:
        # Z 에러 방어 기준(Memory Z)으로 회로 생성
        circuit = evaluator.generate_circuit(Hx, Hz, noise_rate=p, is_x_basis=False) 
        
        task = sinter.Task(
            circuit=circuit,
            json_metadata={'p': p, 'd': 3} 
        )
        tasks.append(task)
        
    if logger:
        logger.info(f"   병렬 워커({multiprocessing.cpu_count()}개)를 기동하여 디코딩을 진행합니다...")
    
    # 2. Sinter 병렬 수집 실행 (최대 10만 샷)
    stats = sinter.collect(
        num_workers=multiprocessing.cpu_count(),
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=100_000,   
        max_errors=1000,     
        print_progress=False
    )
    
    # 3. CSV 데이터 저장
    sinter_dir = os.path.join(save_dir, "sinter_results")
    os.makedirs(sinter_dir, exist_ok=True)
    
    csv_path = os.path.join(sinter_dir, "threshold_data.csv")
    xs, ys = [], []
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Physical_Error_Rate(p)', 'Logical_Error_Rate(P_L)', 'Errors', 'Shots'])
        
        for stat in stats:
            p = stat.json_metadata['p']
            logical_err_rate = stat.errors / stat.shots if stat.shots > 0 else 0.0
            
            writer.writerow([p, logical_err_rate, stat.errors, stat.shots])
            xs.append(p)
            ys.append(logical_err_rate)
            
    if logger:
        logger.info(f"   ✅ [Sinter] CSV 데이터 저장 완료: {csv_path}")
    
    # 4. 논문용 Threshold 그래프(Figure) 생성
    plt.figure(figsize=(8, 6))
    
    plt.plot(xs, ys, marker='o', color='blue', linewidth=2, label='AI Generated Code')
    plt.plot(xs, xs, linestyle='--', color='gray', label='Break-even (Physical = Logical)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate (p)', fontsize=12)
    plt.ylabel('Logical Error Rate ($P_L$)', fontsize=12)
    plt.title('QEC Performance: Threshold Evaluation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plot_path = os.path.join(sinter_dir, "threshold_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"   ✅ [Sinter] Threshold Plot 생성 완료: {plot_path}")