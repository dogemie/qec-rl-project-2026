import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_surface_code_style(Hx, Hz, save_dir, filename_prefix):
    """
    Hx, Hz 패리티 행렬을 Surface Code 스타일(노란색 X, 초록색 Z)의 태너 그래프로 시각화하여
    PNG와 SVG 형식으로 모두 저장합니다.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_qubits = Hx.shape[1]
    m_x = Hx.shape[0]
    m_z = Hz.shape[0]
    
    G = nx.Graph()
    
    # 1. 노드 추가 (데이터 큐비트, X 안정자, Z 안정자)
    data_nodes = [f"D{i}" for i in range(n_qubits)]
    x_nodes = [f"X{i}" for i in range(m_x)]
    z_nodes = [f"Z{i}" for i in range(m_z)]
    
    G.add_nodes_from(data_nodes, bipartite=0)
    G.add_nodes_from(x_nodes, bipartite=1)
    G.add_nodes_from(z_nodes, bipartite=1)
    
    # 2. 엣지 추가 (행렬 값이 1인 곳을 연결)
    for i in range(m_x):
        for j in range(n_qubits):
            if Hx[i, j] == 1:
                G.add_edge(f"X{i}", f"D{j}")
                
    for i in range(m_z):
        for j in range(n_qubits):
            if Hz[i, j] == 1:
                G.add_edge(f"Z{i}", f"D{j}")

    # 3. 그래프 레이아웃 설정 (Spring Layout으로 유기적인 형태 구성)
    for node in x_nodes: G.nodes[node]['layer'] = 0     # 1층: X 안정자
    for node in data_nodes: G.nodes[node]['layer'] = 1  # 2층: 데이터 큐비트
    for node in z_nodes: G.nodes[node]['layer'] = 2     # 3층: Z 안정자
    pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')
    
    # 4. 그리기 (Surface Code 색상 테마 적용)
    plt.figure(figsize=(12, 8))
    
    # 간선 그리기
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="gray")
    
    # 데이터 큐비트 (하얀 바탕에 검은 테두리 점)
    nx.draw_networkx_nodes(G, pos, nodelist=data_nodes, node_color='white', 
                           edgecolors='black', node_size=600, label="Data Qubit")
    # X 안정자 (노란색)
    nx.draw_networkx_nodes(G, pos, nodelist=x_nodes, node_color='gold', 
                           edgecolors='black', node_size=800, node_shape='s', label="X Stabilizer")
    # Z 안정자 (초록색)
    nx.draw_networkx_nodes(G, pos, nodelist=z_nodes, node_color='limegreen', 
                           edgecolors='black', node_size=800, node_shape='s', label="Z Stabilizer")
    
    # 라벨(이름) 표시
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    plt.title("QEC Code Structure (Tanner Graph Representation)", fontsize=16)
    plt.legend(scatterpoints=1, loc='upper right')
    plt.axis('off')
    
    # 5. PNG와 SVG로 각각 저장
    png_path = os.path.join(save_dir, f"{filename_prefix}.png")
    svg_path = os.path.join(save_dir, f"{filename_prefix}.svg")
    
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    
    print(f"📊 그래프가 저장되었습니다: \n - {png_path}\n - {svg_path}")