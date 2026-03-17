import numpy as np
import stim
import pymatching
import sympy
import os

class StimEvaluator:
    def __init__(self, num_qubits, noise_rate=0.01):
        self.n = num_qubits
        self.p = noise_rate

    def _get_gf2_nullspace(self, matrix):
        M = sympy.Matrix(matrix)
        nullspace_basis = M.nullspace()
        return [(np.array(vec).flatten() % 2).astype(np.int8) for vec in nullspace_basis]

    def _gf2_rank(self, matrix):
        if len(matrix) == 0: return 0
        M = sympy.Matrix(matrix)
        _, pivot_cols = M.rref(iszerofunc=lambda x: x % 2 == 0)
        return len(pivot_cols)

    def _find_logical_operators(self, Hx, Hz):
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

    # 🌟 [NEW] 회로 생성 로직만 순수하게 분리 (Sinter와 공유하기 위함)
    def _build_circuit(self, H_measure, L_obs, noise_prob, is_x_basis=False):
        m = H_measure.shape[0]
        circuit = stim.Circuit()

        circuit.append("R", range(self.n + m))
        
        if is_x_basis:
            circuit.append("H", range(self.n))

        circuit.append("DEPOLARIZE1", range(self.n), noise_prob)

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

    def _simulate_code_capacity(self, H_measure, L_obs, noise_prob, is_x_basis=False, num_shots=10000):
        # 분리된 회로 생성기 호출
        circuit = self._build_circuit(H_measure, L_obs, noise_prob, is_x_basis)

        sampler = circuit.compile_detector_sampler()
        syndromes, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
        
        matcher = pymatching.Matching.from_stim_circuit(circuit)
        predicted_observables = matcher.decode_batch(syndromes)
        
        num_errors = np.sum(predicted_observables != actual_observables)
        error_rate = num_errors / num_shots
        
        return error_rate, circuit

    # 🌟 [NEW] Sinter가 호출할 범용 회로 생성 함수
    def generate_circuit(self, Hx, Hz, noise_rate, is_x_basis=False):
        """Hx, Hz 행렬을 바탕으로 특정 노이즈 환경의 stim.Circuit 객체를 반환합니다."""
        Lx, Lz = self._find_logical_operators(Hx, Hz)
        if Lx is None or Lz is None:
            return stim.Circuit() 

        if is_x_basis:
            return self._build_circuit(Hz, Lx, noise_rate, is_x_basis=True)
        else:
            return self._build_circuit(Hx, Lz, noise_rate, is_x_basis=False)

    def evaluate_logical_error_rate(self, Hx, Hz, noise_rate=None, num_shots=10000):
        if noise_rate is None:
            noise_rate = self.p
            
        Lx, Lz = self._find_logical_operators(Hx, Hz)
        if Lx is None or Lz is None:
            return 1.0 

        z_error_rate, z_circuit = self._simulate_code_capacity(Hx, Lz, noise_rate, False, num_shots)
        x_error_rate, x_circuit = self._simulate_code_capacity(Hz, Lx, noise_rate, True, num_shots)

        worst_error_rate = max(z_error_rate, x_error_rate)
        self._last_valid_circuit = z_circuit 
        
        return worst_error_rate

    def save_circuit_diagram(self, Hx, Hz, save_dir, filename="circuit_timeline.svg"):
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(self, '_last_valid_circuit') and self._last_valid_circuit is not None:
            file_path = os.path.join(save_dir, filename)
            svg_content = self._last_valid_circuit.diagram("timeline-svg")
            svg_content = str(svg_content).replace('<svg ', '<svg style="background-color:white;" ')
            with open(file_path, "w") as f:
                print(svg_content, file=f)