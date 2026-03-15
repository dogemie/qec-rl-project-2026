import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_2d_grid_layout(hx_path, hz_path, save_path=None, show_plot=True):
    """
    물리적 2D 격자(Grid) 위에서 QEC 코드의 배선을 시각화합니다.
    AI가 하드웨어 거리를 얼마나 잘 최적화했는지 확인하는 용도입니다.
    """
    if not os.path.exists(hx_path) or not os.path.exists(hz_path):
        print("❌ Hx 또는 Hz numpy 파일을 찾을 수 없습니다.")
        return

    Hx = np.load(hx_path)
    Hz = np.load(hz_path)

    num_x_stabs, num_qubits = Hx.shape
    num_z_stabs = Hz.shape[0]
    total_nodes = num_qubits + num_x_stabs + num_z_stabs

    # AI가 훈련 시 사용한 것과 동일한 Auto-Grid 크기 계산
    grid_size = int(np.ceil(np.sqrt(total_nodes)))

    # 인덱스를 2D 격자 좌표 (x, y)로 변환하는 함수
    # y좌표는 위에서 아래로 내려오도록 마이너스(-) 처리
    def get_coord(index):
        row = index // grid_size
        col = index % grid_size
        return (col, -row)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # 칩의 배경 느낌을 주는 연한 점선 그리드 그리기
    for i in range(grid_size + 1):
        ax.axhline(-i + 0.5, color='whitesmoke', linestyle='--', zorder=0)
        ax.axvline(i - 0.5, color='whitesmoke', linestyle='--', zorder=0)

    total_distance = 0
    total_cnots = 0

    # 1. Hx 배선 그리기 (X 안정자 -> 데이터 큐비트)
    for i in range(num_x_stabs):
        stab_idx = num_qubits + i
        sx, sy = get_coord(stab_idx)
        for j in range(num_qubits):
            if Hx[i, j] == 1:
                qx, qy = get_coord(j)
                total_distance += abs(sx - qx) + abs(sy - qy)
                total_cnots += 1
                ax.plot([sx, qx], [sy, qy], color='gray', linestyle='-', linewidth=1.5, alpha=0.6, zorder=1)

    # 2. Hz 배선 그리기 (Z 안정자 -> 데이터 큐비트)
    for i in range(num_z_stabs):
        stab_idx = num_qubits + num_x_stabs + i
        sx, sy = get_coord(stab_idx)
        for j in range(num_qubits):
            if Hz[i, j] == 1:
                qx, qy = get_coord(j)
                total_distance += abs(sx - qx) + abs(sy - qy)
                total_cnots += 1
                ax.plot([sx, qx], [sy, qy], color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

    # 3. 노드 그리기
    node_radius = 0.3
    for idx in range(total_nodes):
        x, y = get_coord(idx)
        
        if idx < num_qubits:
            # 데이터 큐비트 (회색 원)
            circle = patches.Circle((x, y), node_radius, facecolor='lightgray', edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, f'D{idx}', ha='center', va='center', fontweight='bold', fontsize=12)
        elif idx < num_qubits + num_x_stabs:
            # X 안정자 (노란색 사각형)
            stab_num = idx - num_qubits
            rect = patches.Rectangle((x - node_radius, y - node_radius), node_radius*2, node_radius*2, 
                                     facecolor='gold', edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(rect)
            ax.text(x, y, f'X{stab_num}', ha='center', va='center', fontweight='bold', fontsize=12)
        else:
            # Z 안정자 (초록색 사각형)
            stab_num = idx - num_qubits - num_x_stabs
            rect = patches.Rectangle((x - node_radius, y - node_radius), node_radius*2, node_radius*2, 
                                     facecolor='limegreen', edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(rect)
            ax.text(x, y, f'Z{stab_num}', ha='center', va='center', fontweight='bold', fontsize=12)

    # 타이틀 및 메타데이터 표시
    plt.title("AI Generated QEC Code on 2D Hardware Grid", fontsize=16, fontweight='bold', pad=20)
    info_text = f"Total CNOTs: {total_cnots}\nTotal Manhattan Distance: {total_distance}"
    plt.figtext(0.15, 0.15, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    ax.axis('off')
    
    # 여백 조절
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-grid_size + 0.5, 0.5)

    if save_path:
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight", facecolor='white')
        print(f"✅ 2D Grid 이미지가 저장되었습니다: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# 독립 실행을 위한 테스트 코드
if __name__ == "__main__":
    # 방금 전 기록을 달성한 폴더의 경로를 입력해 테스트해 볼 수 있습니다.
    # 예시:
    # hx_file = "../outputs/260315_2151_s53832/best_codes_epcX_epY/best_Hx.npy"
    # hz_file = "../outputs/260315_2151_s53832/best_codes_epcX_epY/best_Hz.npy"
    # draw_2d_grid_layout(hx_file, hz_file, save_path="test_2d_grid.png")
    pass