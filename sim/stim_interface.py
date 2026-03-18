import numpy as np
import stim
import pymatching
import sympy
import os

class StimEvaluator:
    """
    QEC 회로의 에러 억제 성능을 시뮬레이션 및 평가하는 모듈.
    """
    def __init__(self, num_qubits, noise_rate_x=0.01, noise_rate_z=0.01):
        """
        StimEvaluator를 초기화합니다. 편향 노이즈(Biased noise) 테스트를 위해
        X 에러와 Z 에러 확률을 별도로 설정할 수 있습니다.
        
        Args:
            num_qubits (int): 시스템 내 데이터 큐비트의 수.
            noise_rate_x (float): 비트 플립(X 에러)이 발생할 확률.
            noise_rate_z (float): 위상 플립(Z 에러)이 발생할 확률.
        """
        self.n = num_qubits
        self.p_x = noise_rate_x
        self.p_z = noise_rate_z

    def _get_gf2_nullspace(self, matrix):
        """
        주어진 이진 행렬의 GF(2) 상 영공간(Nullspace) 기저를 반환합니다.
        
        Args:
            matrix (np.ndarray): 분석할 이진 행렬.
            
        Returns:
            list: 영공간을 구성하는 기저 벡터 리스트.
        """
        M = sympy.Matrix(matrix)
        nullspace_basis = M.nullspace()
        return [(np.array(vec).flatten() % 2).astype(np.int8) for vec in nullspace_basis]

    def _gf2_rank(self, matrix):
        """
        주어진 이진 행렬의 GF(2) 상 계수(Rank)를 계산합니다.
        
        Args:
            matrix (np.ndarray): 랭크를 계산할 이진 행렬.
            
        Returns:
            int: 행렬의 계수(선형 독립인 행/열의 개수).
        """
        if len(matrix) == 0: return 0
        M = sympy.Matrix(matrix)
        _, pivot_cols = M.rref(iszerofunc=lambda x: x % 2 == 0)
        return len(pivot_cols)

    def _find_logical_operators(self, Hx, Hz):
        """
        주어진 안정자 행렬(Hx, Hz)을 기반으로 유효한 논리적 관측값(Lx, Lz) 쌍을 탐색합니다.
        
        Args:
            Hx (np.ndarray): X 안정자 행렬.
            Hz (np.ndarray): Z 안정자 행렬.
            
        Returns:
            tuple: 논리적 X 연산자와 Z 연산자 벡터. 조건을 만족하는 쌍이 없으면 (None, None).
        """
        Lz_candidates = self._get_gf2_nullspace(Hx)
        Lx_candidates = self._get_gf2_nullspace(Hz)
        rank_Hz = self._gf2_rank(Hz)
        rank_Hx = self._gf2_rank(Hx)

        valid_Lz = []
        for lz in Lz_candidates:
            if np.all(lz == 0): continue
            if self._gf2_rank(np.vstack([Hz, lz])) > rank_Hz:
                valid_Lz.append(lz)

        valid_Lx = []
        for lx in Lx_candidates:
            if np.all(lx == 0): continue
            if self._gf2_rank(np.vstack([Hx, lx])) > rank_Hx:
                valid_Lx.append(lx)

        for lx in valid_Lx:
            for lz in valid_Lz:
                if np.dot(lx, lz) % 2 == 1:
                    return lx, lz 
        return None, None

    def _build_circuit(self, H_measure, L_obs, p_x, p_z, is_x_basis=False):
        """
        주어진 행렬 정보와 노이즈 확률을 기반으로 stim.Circuit 객체를 생성합니다.
        
        Args:
            H_measure (np.ndarray): 측정을 수행할 안정자 행렬.
            L_obs (np.ndarray): 목표로 하는 논리적 관측값 연산자.
            p_x (float): 데이터 큐비트에 적용할 X 에러 발생 확률.
            p_z (float): 데이터 큐비트에 적용할 Z 에러 발생 확률.
            is_x_basis (bool): X 기저에서 측정할 경우 True, Z 기저인 경우 False.
            
        Returns:
            stim.Circuit: 노이즈와 신드롬 측정 절차가 포함된 양자 회로.
        """
        m = H_measure.shape[0]
        circuit = stim.Circuit()

        circuit.append("R", range(self.n + m))
        
        if is_x_basis:
            circuit.append("H", range(self.n))

        if p_x > 0:
            circuit.append("X_ERROR", range(self.n), p_x)
        if p_z > 0:
            circuit.append("Z_ERROR", range(self.n), p_z)

        for i, row in enumerate(H_measure):
            ancilla_idx = self.n + i
            if is_x_basis:
                circuit.append("H", [ancilla_idx])
                
            for qubit_idx, val in enumerate(row):
                if val == 1:
                    if is_x_basis:
                        circuit.append("CX", [ancilla_idx, qubit_idx])
                    else:
                        circuit.append("CX", [qubit_idx, ancilla_idx])
                        
            if is_x_basis:
                circuit.append("H", [ancilla_idx])
                
            circuit.append("M", [ancilla_idx])
            circuit.append("DETECTOR", [stim.target_rec(-1)])

        if is_x_basis:
            circuit.append("H", range(self.n))
        circuit.append("M", range(self.n))

        obs_targets = [stim.target_rec(-self.n + q) for q, val in enumerate(L_obs) if val == 1]
        circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)
        
        return circuit

    def _simulate_code_capacity(self, H_measure, L_obs, p_x, p_z, is_x_basis=False, num_shots=10000):
        """
        단일 기저에 대한 회로를 생성하고, PyMatching을 통해 디코딩 시뮬레이션을 수행합니다.
        
        Args:
            H_measure (np.ndarray): 측정을 수행할 안정자 행렬.
            L_obs (np.ndarray): 목표로 하는 논리적 관측값 연산자.
            p_x (float): 주입할 X 에러 확률.
            p_z (float): 주입할 Z 에러 확률.
            is_x_basis (bool): 평가 기저 설정.
            num_shots (int): 몬테카를로 시뮬레이션 반복 횟수.
            
        Returns:
            tuple: (계산된 논리적 에러율, 사용된 stim.Circuit 객체)
        """
        circuit = self._build_circuit(H_measure, L_obs, p_x, p_z, is_x_basis)

        sampler = circuit.compile_detector_sampler()
        syndromes, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
        
        matcher = pymatching.Matching.from_stim_circuit(circuit)
        predicted_observables = matcher.decode_batch(syndromes)
        
        num_errors = np.sum(predicted_observables != actual_observables)
        error_rate = num_errors / num_shots
        
        return error_rate, circuit

    def generate_circuit(self, Hx, Hz, noise_rate_x=None, noise_rate_z=None, is_x_basis=False):
        """
        행렬 및 노이즈 설정을 바탕으로 범용적으로 활용 가능한 stim.Circuit 객체를 생성하여 반환합니다.
        
        Args:
            Hx (np.ndarray): 시스템의 X 안정자 행렬.
            Hz (np.ndarray): 시스템의 Z 안정자 행렬.
            noise_rate_x (float, optional): 적용할 X 노이즈 확률.
            noise_rate_z (float, optional): 적용할 Z 노이즈 확률.
            is_x_basis (bool): X 기저 회로를 생성할지 여부.
            
        Returns:
            stim.Circuit: Sinter 등 외부 도구에서 평가할 수 있는 양자 회로 객체.
        """
        if noise_rate_x is None: noise_rate_x = self.p_x
        if noise_rate_z is None: noise_rate_z = self.p_z
        
        Lx, Lz = self._find_logical_operators(Hx, Hz)
        if Lx is None or Lz is None:
            return stim.Circuit() 

        if is_x_basis:
            return self._build_circuit(Hz, Lx, noise_rate_x, noise_rate_z, is_x_basis=True)
        else:
            return self._build_circuit(Hx, Lz, noise_rate_x, noise_rate_z, is_x_basis=False)

    def evaluate_logical_error_rate(self, Hx, Hz, noise_rate_x=None, noise_rate_z=None, num_shots=10000):
        """
        생성된 QEC 코드가 논리적 에러를 얼마나 효과적으로 방어하는지 종합 평가합니다.
        가장 취약한(에러율이 높은) 기저의 결과를 기준점수로 채택합니다.
        
        Args:
            Hx (np.ndarray): X 안정자 행렬.
            Hz (np.ndarray): Z 안정자 행렬.
            noise_rate_x (float, optional): 주입할 X 에러 확률. 미지정 시 초기화 값 사용.
            noise_rate_z (float, optional): 주입할 Z 에러 확률. 미지정 시 초기화 값 사용.
            num_shots (int): 시뮬레이션 반복 횟수.
            
        Returns:
            float: X 기저와 Z 기저 중 더 높은(성능이 떨어지는) 논리적 에러율.
        """
        if noise_rate_x is None: noise_rate_x = self.p_x
        if noise_rate_z is None: noise_rate_z = self.p_z
            
        Lx, Lz = self._find_logical_operators(Hx, Hz)
        if Lx is None or Lz is None:
            return 1.0 

        z_error_rate, z_circuit = self._simulate_code_capacity(Hx, Lz, noise_rate_x, noise_rate_z, False, num_shots)
        x_error_rate, x_circuit = self._simulate_code_capacity(Hz, Lx, noise_rate_x, noise_rate_z, True, num_shots)

        worst_error_rate = max(z_error_rate, x_error_rate)
        self._last_valid_circuit = z_circuit 
        
        return worst_error_rate

    def save_circuit_diagram(self, Hx, Hz, save_dir, filename="circuit_timeline.svg"):
        """
        최근 생성된 유효 회로를 SVG 다이어그램 이미지 파일로 저장합니다.
        
        Args:
            Hx (np.ndarray): 회로 생성의 기초가 된 X 안정자. (현재 버전에서는 내부 캐시 활용)
            Hz (np.ndarray): 회로 생성의 기초가 된 Z 안정자.
            save_dir (str): 이미지를 저장할 디렉토리 경로.
            filename (str): 생성될 파일의 이름.
        """
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(self, '_last_valid_circuit') and self._last_valid_circuit is not None:
            file_path = os.path.join(save_dir, filename)
            svg_content = self._last_valid_circuit.diagram("timeline-svg")
            svg_content = str(svg_content).replace('<svg ', '<svg style="background-color:white;" ')
            with open(file_path, "w") as f:
                print(svg_content, file=f)