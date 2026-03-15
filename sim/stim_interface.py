import numpy as np
import stim
import pymatching
import sympy
import os

class StimEvaluator:
    """
    AI가 생성한 QEC 패리티 행렬(Hx, Hz)을 평가하여 논리적 에러율(P_L)을 반환하는 엄격한 심판 모듈.
    이제 반쪽짜리 방어(꼼수)는 허용하지 않으며, 실제 물리적 노이즈 환경과 동일한 엄격한 기준을 적용합니다.
    """
    def __init__(self, num_qubits, noise_rate=0.01):
        self.n = num_qubits
        self.p = noise_rate  # 물리적 큐비트의 에러 발생 확률 (예: 1%)

    def _get_gf2_nullspace(self, matrix):
        """주어진 행렬의 GF(2) 이진 영공간(Nullspace) 기저를 찾습니다."""
        M = sympy.Matrix(matrix)
        nullspace_basis = M.nullspace()
        return [(np.array(vec).flatten() % 2).astype(np.int8) for vec in nullspace_basis]

    def _gf2_rank(self, matrix):
        """GF(2) 상에서 행렬의 랭크(선형 독립인 행의 개수)를 계산합니다."""
        if len(matrix) == 0: return 0
        M = sympy.Matrix(matrix)
        _, pivot_cols = M.rref(iszerofunc=lambda x: x % 2 == 0)
        return len(pivot_cols)

    def _find_logical_operators(self, Hx, Hz):
        """
        [핵심 평가 1] 주어진 Hx, Hz 패리티 행렬에서 완벽한 논리 큐비트 한 쌍(L_X, L_Z)을 찾습니다.
        단순히 교환 법칙만 만족하는 것이 아니라, 다음 세 가지 조건을 모두 만족해야 합니다:
        1. L_Z는 Hx와 교환되어야 하고, Hz(안정자)에 속하지 않아야 함.
        2. L_X는 Hz와 교환되어야 하고, Hx(안정자)에 속하지 않아야 함.
        3. 🌟 (가장 중요) L_X와 L_Z는 서로 반교환(Anti-commute)해야 함! (내적 결과가 홀수)
        """
        Lz_candidates = self._get_gf2_nullspace(Hx)
        Lx_candidates = self._get_gf2_nullspace(Hz)
        
        rank_Hz = self._gf2_rank(Hz)
        rank_Hx = self._gf2_rank(Hx)

        # 1. Hz의 선형 조합으로 만들어지지 않는 진짜 L_Z 후보 추리기
        valid_Lz = []
        for lz in Lz_candidates:
            if np.all(lz == 0): continue
            if self._gf2_rank(np.vstack([Hz, lz])) > rank_Hz:
                valid_Lz.append(lz)

        # 2. Hx의 선형 조합으로 만들어지지 않는 진짜 L_X 후보 추리기
        valid_Lx = []
        for lx in Lx_candidates:
            if np.all(lx == 0): continue
            if self._gf2_rank(np.vstack([Hx, lx])) > rank_Hx:
                valid_Lx.append(lx)

        # 3. 서로 반교환(내적이 홀수)하는 영혼의 단짝(Anti-commuting pair) 찾기
        for lx in valid_Lx:
            for lz in valid_Lz:
                if np.dot(lx, lz) % 2 == 1:
                    return lx, lz  # 진정한 논리 큐비트 한 쌍 발견!

        # 조건을 만족하는 쌍을 찾지 못하면 가차 없이 탈락 (None 반환)
        return None, None

    def _simulate_code_capacity(self, H_measure, L_obs, noise_prob, is_x_basis=False, num_shots=10000):
        """
        [핵심 평가 2] 단일 기저(X 또는 Z)에 대한 에러 정정 능력을 시뮬레이션합니다.
        현실적인 노이즈를 주입하고, PyMatching을 이용해 에러를 얼마나 잘 추적/복구하는지 테스트합니다.
        """
        m = H_measure.shape[0]
        circuit = stim.Circuit()

        # 1. 초기화 (데이터 큐비트 + 앙실라 큐비트)
        circuit.append("R", range(self.n + m))
        
        if is_x_basis:
            # X 기저 테스트 시 데이터 큐비트를 |+> 상태로 초기화
            circuit.append("H", range(self.n))

        # 2. 노이즈 주입 (Depolarizing Noise: X, Y, Z 에러가 모두 발생할 수 있는 현실적 환경)
        circuit.append("DEPOLARIZE1", range(self.n), noise_prob)

        # 3. 신드롬 측정 (안정자 행렬에 기반한 얽힘 및 측정)
        for i, row in enumerate(H_measure):
            ancilla_idx = self.n + i
            if is_x_basis:
                circuit.append("H", [ancilla_idx]) # X 안정자는 앙실라를 |+>로 세팅
                
            for qubit_idx, val in enumerate(row):
                if val == 1:
                    if is_x_basis:
                        # X 안정자: 앙실라가 Control, 데이터가 Target
                        circuit.append("CX", [ancilla_idx, qubit_idx])
                    else:
                        # Z 안정자: 데이터가 Control, 앙실라가 Target
                        circuit.append("CX", [qubit_idx, ancilla_idx])
                        
            if is_x_basis:
                circuit.append("H", [ancilla_idx])
                
            # 앙실라 측정 및 검출기 선언
            circuit.append("M", [ancilla_idx])
            circuit.append("DETECTOR", [stim.target_rec(-1)])

        # 4. 데이터 큐비트 최종 측정
        if is_x_basis:
            circuit.append("H", range(self.n))
        circuit.append("M", range(self.n))

        # 5. 논리적 관측값(Observable) 선언 (에러 정정의 성공 여부를 판가름할 기준)
        obs_targets = [stim.target_rec(-self.n + q) for q, val in enumerate(L_obs) if val == 1]
        circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)

        # 6. PyMatching으로 디코딩 및 실제 에러율 계산
        sampler = circuit.compile_detector_sampler()
        syndromes, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
        
        matcher = pymatching.Matching.from_stim_circuit(circuit)
        predicted_observables = matcher.decode_batch(syndromes)
        
        num_errors = np.sum(predicted_observables != actual_observables)
        error_rate = num_errors / num_shots
        
        return error_rate, circuit

    def evaluate_logical_error_rate(self, Hx, Hz, num_shots=10000):
        """
        [최종 채점] 생성된 회로가 X 에러와 Z 에러를 모두 방어하는지 엄격하게 종합 채점합니다.
        가장 취약한 부분의 에러율을 최종 점수로 반환하여 AI의 꼼수를 차단합니다.
        """
        Lx, Lz = self._find_logical_operators(Hx, Hz)
        
        if Lx is None or Lz is None:
            # 진정한 양자 코드의 조건을 갖추지 못했으면 최악의 에러율(1.0) 반환
            return 1.0 

        # 테스트 1: Z 에러 방어력 검증 (Hx로 측정, 논리적 Z 보존 여부 확인)
        z_error_rate, z_circuit = self._simulate_code_capacity(
            H_measure=Hx, L_obs=Lz, noise_prob=self.p, is_x_basis=False, num_shots=num_shots
        )
        
        # 테스트 2: X 에러 방어력 검증 (Hz로 측정, 논리적 X 보존 여부 확인)
        x_error_rate, x_circuit = self._simulate_code_capacity(
            H_measure=Hz, L_obs=Lx, noise_prob=self.p, is_x_basis=True, num_shots=num_shots
        )

        # 🌟 가장 취약한 부분을 최종 에러율로 산정 (방패의 양면이 모두 튼튼해야 함)
        worst_error_rate = max(z_error_rate, x_error_rate)
        
        # 시각화를 위해 기본 Z_circuit을 캐싱해 둡니다.
        self._last_valid_circuit = z_circuit 
        
        return worst_error_rate

    def save_circuit_diagram(self, Hx, Hz, save_dir, filename="circuit_timeline.svg"):
        """
        생성된 Stim 회로를 시각적으로 알아보기 쉬운 SVG 이미지 파일로 저장합니다.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 캐싱된 회로가 있는지 확인 (evaluate_logical_error_rate가 먼저 실행되어야 함)
        if hasattr(self, '_last_valid_circuit') and self._last_valid_circuit is not None:
            file_path = os.path.join(save_dir, filename)
            
            # Stim에서 SVG 문자열을 뽑아냄
            svg_content = self._last_valid_circuit.diagram("timeline-svg")
            
            # 다크 모드 환경에서도 잘 보이도록 최상단 <svg> 태그에 흰색 배경 스타일 강제 주입
            svg_content = str(svg_content).replace('<svg ', '<svg style="background-color:white;" ')
            
            # timeline-svg 포맷으로 회로도를 생성하여 저장
            with open(file_path, "w") as f:
                print(self._last_valid_circuit.diagram("timeline-svg"), file=f)
                
            print(f"회로도가 {file_path} 파일로 성공적으로 저장되었습니다!")
        else:
            print("저장할 수 있는 유효한 회로가 없습니다. (에러율 평가를 통과하지 못함)")