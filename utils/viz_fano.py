import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_fano_steane_graph(hx_path, hz_path, save_path=None, show_plot=False):
    # 1. 행렬 데이터 불러오기
    Hx = np.load(hx_path)
    Hz = np.load(hz_path)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 2. 교과서(Fano Plane) 이미지 기반의 고정 좌표 설정
    # 데이터 큐비트 (회색 원) - 7개
    d_pos = {
        0: (0, 4),           # p1 (Top)
        1: (-3.46, -2),      # p2 (Bottom Left)
        2: (3.46, -2),       # p3 (Bottom Right)
        3: (-1.73, 1),       # d1 (Mid Left)
        4: (1.73, 1),        # d2 (Mid Right)
        5: (0, -2),          # d3 (Mid Bottom)
        6: (0, 0)            # d4 (Center)
    }
    
    # X 안정자 (노란색 사각형) - 3개 (이미지의 s1, s2, s3 위치)
    x_pos = {
        0: (-0.6, 2.0),      # Top Left
        1: (-1.5, -1.0),     # Bottom Inner Left
        2: (1.5, -1.0)       # Bottom Inner Right
    }
    
    # Z 안정자 (초록색 사각형) - 3개 (이미지의 s4, s5, s6 위치)
    z_pos = {
        0: (0.6, 2.0),       # Top Right
        1: (-2.5, -0.2),     # Outer Left
        2: (2.5, -0.2)       # Outer Right
    }

    # 3. 연결선(Edges) 그리기
    # X 안정자 연결선 (실선)
    for i in range(Hx.shape[0]):
        for j in range(Hx.shape[1]):
            if Hx[i, j] == 1:
                ax.plot([x_pos[i][0], d_pos[j][0]], [x_pos[i][1], d_pos[j][1]], 
                        color='gray', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)
                
    # Z 안정자 연결선 (점선)
    for i in range(Hz.shape[0]):
        for j in range(Hz.shape[1]):
            if Hz[i, j] == 1:
                ax.plot([z_pos[i][0], d_pos[j][0]], [z_pos[i][1], d_pos[j][1]], 
                        color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

    # 4. 노드(Nodes) 그리기
    # 데이터 큐비트
    for idx, (x, y) in d_pos.items():
        circle = patches.Circle((x, y), 0.4, facecolor='lightgray', edgecolor='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, f'D{idx}', fontsize=12, fontweight='bold', ha='center', va='center', zorder=3)

    # X 안정자
    for idx, (x, y) in x_pos.items():
        rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, facecolor='gold', edgecolor='black', linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, f'X{idx}', fontsize=12, fontweight='bold', ha='center', va='center', zorder=3)

    # Z 안정자
    for idx, (x, y) in z_pos.items():
        rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, facecolor='limegreen', edgecolor='black', linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, f'Z{idx}', fontsize=12, fontweight='bold', ha='center', va='center', zorder=3)

    plt.title("AI Generated QEC Code on Steane Fano Plane Layout", fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight", facecolor='white')
        print(f"이미지가 저장되었습니다: {save_path}")
    
    if show_plot: # 🌟 수정: show_plot이 True일 때만 화면에 띄움
        plt.show()
    else:
        plt.close(fig) # 메모리 누수 방지

# 🌟 실행 예시 (경로를 회원님의 실제 파일 경로로 수정하세요)
# Hx_path = "outputs/260315_1214_s49105/best_codes_epc27_ep16/best_Hx.npy"
# Hz_path = "outputs/260315_1214_s49105/best_codes_epc27_ep16/best_Hz.npy"
# draw_fano_steane_graph(Hx_path, Hz_path, save_path="fano_comparison.png, show_plot=True)")