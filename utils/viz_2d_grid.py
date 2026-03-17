import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_2d_grid_layout(hx_path, hz_path, save_path=None, show_plot=True):
    """
    물리적 2D 격자(Grid) 위에서 QEC 코드의 배선을 시각화합니다.
    (범용 동적 격자 확장 및 패리티 기반 매핑 적용)
    """
    if not os.path.exists(hx_path) or not os.path.exists(hz_path):
        print("❌ Hx 또는 Hz numpy 파일을 찾을 수 없습니다.")
        return

    Hx = np.load(hx_path)
    Hz = np.load(hz_path)

    num_x_stabs, num_qubits = Hx.shape
    num_z_stabs = Hz.shape[0]
    total_nodes = num_qubits + num_x_stabs + num_z_stabs

    # 🌟 회원님의 일반화 공식 적용: ceil(sqrt(qubits)) * 2 + 5
    n = int(np.ceil(np.sqrt(num_qubits)))
    grid_size = n * 2 + 5

    def get_coord(index):
        if not hasattr(get_coord, "mapping"):
            mapping = {}
            center = grid_size // 2
            data_coords = []
            stab_coords = []
            
            # 1. 격자를 돌면서 중심을 기준으로 짝수/홀수 칸 분류
            for y in range(grid_size):
                for x in range(grid_size):
                    # 중심으로부터의 거리 (정사각형 형태로 퍼져나가도록 Chebyshev 거리 사용)
                    dist = max(abs(x - center), abs(y - center))
                    
                    # 데이터 큐비트는 무조건 홀수 좌표 (x, y 모두 홀수)
                    if x % 2 == 1 and y % 2 == 1:
                        data_coords.append((dist, y, x, x, -y))
                    else:
                        # 안정자는 그 외의 공간
                        stab_coords.append((dist, y, x, x, -y))
                        
            # 2. 중심에서 가까운 순서대로 정렬 (맵 중앙에 뭉치도록 유도)
            data_coords.sort()
            stab_coords.sort()
            
            # 정렬 후 실제 좌표(x, -y)만 추출
            data_coords = [(x, y) for _, _, _, x, y in data_coords]
            stab_coords = [(x, y) for _, _, _, x, y in stab_coords]
            
            # 3. 인덱스에 좌표 할당
            for i in range(total_nodes):
                if i < num_qubits:
                    mapping[i] = data_coords.pop(0) if data_coords else stab_coords.pop(0)
                else:
                    mapping[i] = stab_coords.pop(0) if stab_coords else data_coords.pop(0)
                    
            get_coord.mapping = mapping

        return get_coord.mapping[index]

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
            circle = patches.Circle((x, y), node_radius, facecolor='lightgray', edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, f'D{idx}', ha='center', va='center', fontweight='bold', fontsize=12)
        elif idx < num_qubits + num_x_stabs:
            stab_num = idx - num_qubits
            rect = patches.Rectangle((x - node_radius, y - node_radius), node_radius*2, node_radius*2, 
                                     facecolor='gold', edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(rect)
            ax.text(x, y, f'X{stab_num}', ha='center', va='center', fontweight='bold', fontsize=12)
        else:
            stab_num = idx - num_qubits - num_x_stabs
            rect = patches.Rectangle((x - node_radius, y - node_radius), node_radius*2, node_radius*2, 
                                     facecolor='limegreen', edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(rect)
            ax.text(x, y, f'Z{stab_num}', ha='center', va='center', fontweight='bold', fontsize=12)

    plt.title(f"Hardware Grid Layout (Size: {grid_size}x{grid_size})", fontsize=16, fontweight='bold', pad=20)
    info_text = f"Total CNOTs: {total_cnots}\nTotal Manhattan Distance: {total_distance}"
    plt.figtext(0.15, 0.15, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    ax.axis('off')
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-grid_size + 0.5, 0.5)

    if save_path:
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight", facecolor='white')
        print(f"✅ 2D Grid 이미지가 저장되었습니다: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)