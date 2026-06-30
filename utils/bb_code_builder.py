import numpy as np

class BBCodeBuilder:
    """
    Bivariate Bicycle (BB) 기반의 QLDPC 코드를 생성하는 수학 엔진.
    AI가 선택한 다항식 시프트(Shift) 배열을 받아, Sinter가 이해할 수 있는 
    거대한 H_X, H_Z 패리티 검사 행렬로 전개(Expansion)합니다.
    """
    def __init__(self, L):
        """
        Args:
            L (int): 블록 행렬의 크기 (Block Size). 
                     데이터 큐비트의 수는 2*L, X/Z 안정자의 수는 각각 L개가 됩니다.
        """
        self.L = L

    def _get_cyclic_shift_matrix(self, shift_val):
        """
        크기가 L x L인 단위 행렬을 shift_val 만큼 오른쪽으로 순환 이동(Cyclic Shift)시킵니다.
        예: shift_val=1 이면, 대각선(1)이 우측으로 한 칸 이동합니다.
        """
        I = np.eye(self.L, dtype=np.int8)
        # np.roll을 사용하여 행렬의 열을 순환 이동시킵니다.
        return np.roll(I, shift_val, axis=1)

    def build_matrices(self, a1, a2, b1, b2):
        """
        AI가 선택한 4개의 시프트 값을 바탕으로 H_X와 H_Z 행렬을 조립합니다.
        
        Args:
            a1, a2 (int): 행렬 A를 구성하는 시프트 값
            b1, b2 (int): 행렬 B를 구성하는 시프트 값
            
        Returns:
            Hx (np.ndarray): X 안정자 행렬 (크기: L x 2L)
            Hz (np.ndarray): Z 안정자 행렬 (크기: L x 2L)
        """
        # 1. 시프트 값을 기반으로 부분 행렬 A, B 생성
        # A = x^a1 + x^a2  (단, 같은 칸에 겹치면 GF(2) 연산이므로 0이 됨)
        A = (self._get_cyclic_shift_matrix(a1) + self._get_cyclic_shift_matrix(a2)) % 2
        
        # B = y^b1 + y^b2
        B = (self._get_cyclic_shift_matrix(b1) + self._get_cyclic_shift_matrix(b2)) % 2
        
        # 2. BB 코드의 대칭성을 이용한 H_X, H_Z 조립
        # H_X = [A, B]
        Hx = np.hstack((A, B))
        
        # H_Z = [B^T, A^T]
        Hz = np.hstack((B.T, A.T))
        
        return Hx, Hz

    def check_commutativity(self, Hx, Hz):
        """
        [디버깅 및 검증용] H_X * H_Z^T == 0 (mod 2) 교환 법칙이 성립하는지 확인합니다.
        BB 코드 구조상 이 값은 항상 0이어야 합니다!
        """
        dot_product = np.dot(Hx, Hz.T) % 2
        violations = np.sum(dot_product != 0)
        return violations

# =====================================================================
# 🧪 [테스트 코드] 다양한 엣지 케이스 및 스케일 무결성 테스트
# =====================================================================
if __name__ == "__main__":
    print("🚀 [BBCodeBuilder 검증 테스트] 다양한 시나리오에서의 교환 법칙 무결성 테스트 시작!\n")

    test_cases = [
        {"name": "기본 테스트 (소형)", "L": 15, "shifts": [1, 2, 4, 8]},
        {"name": "논문 실전 스케일 (소수 소수 조합)", "L": 31, "shifts": [3, 7, 15, 23]},
        {"name": "동일 시프트 중복 입력 (모듈로 2 상쇄 확인)", "L": 10, "shifts": [2, 2, 5, 5]},
        {"name": "0(Zero) 시프트 (극단적 엣지 케이스)", "L": 12, "shifts": [0, 0, 0, 0]},
        {"name": "거대 행렬 스트레스 테스트", "L": 127, "shifts": [10, 45, 80, 115]},
    ]

    all_passed = True

    for idx, tc in enumerate(test_cases, 1):
        L = tc["L"]
        shifts = tc["shifts"]
        builder = BBCodeBuilder(L)
        
        Hx, Hz = builder.build_matrices(*shifts)
        violations = builder.check_commutativity(Hx, Hz)

        print(f"[{idx}] {tc['name']}")
        print(f"  - L값 (블록 크기): {L} (전체 큐비트: {2*L}개)")
        print(f"  - 입력 시프트(a1, a2, b1, b2): {shifts}")
        print(f"  - Hx 크기: {Hx.shape}, Hz 크기: {Hz.shape}")

        if violations == 0:
            print(f"  - 결과: ✅ PASS (위반 개수: 0개)\n")
        else:
            print(f"  - 결과: ❌ FAIL (위반 개수: {violations}개)\n")
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("🎉 [최종 결과] 5개 테스트 케이스 ALL PASS!")
        print("어떤 기상천외한 숫자가 들어와도 교환 법칙(Commutativity)이 100% 보장됩니다.")
    else:
        print("🚨 [최종 결과] 일부 테스트에서 위반이 발생했습니다. 코드를 점검해 주세요.")