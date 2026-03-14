import torch
import pytest
from agent.network import QECNet

def test_qecnet_forward_pass():
    """
    QECNet(신경망)이 올바른 형태의 텐서를 입력받고,
    정확한 형태의 정책(Policy)과 가치(Value)를 출력하는지 검증합니다.
    특히, '마스킹(Masking)'이 완벽하게 작동하는지 확인합니다.
    """
    batch_size = 4
    num_qubits = 7
    num_stabilizers = 3
    action_size = 2 * num_stabilizers * num_qubits  # 2 * 3 * 7 = 42개의 경우의 수

    # 1. 에이전트 두뇌(신경망) 초기화
    net = QECNet(num_qubits=num_qubits, num_stabilizers=num_stabilizers)

    # 2. 가짜 입력 데이터 (Dummy State) 생성
    # 형태: [배치 사이즈, 채널(Hx, Hz), 행(m), 열(n)] -> (4, 2, 3, 7)
    dummy_state = torch.zeros((batch_size, 2, num_stabilizers, num_qubits), dtype=torch.float32)

    # 3. 행동 마스크(Action Mask) 생성
    # 기본적으로 모든 행동(1)이 가능하다고 가정하되, 특정 행동만 강제로 금지(0)시켜 봅니다.
    dummy_mask = torch.ones((batch_size, action_size), dtype=torch.float32)
    dummy_mask[:, 0] = 0   # 모든 배치에서 '0번 위치'에 비트를 두는 것을 물리적으로 금지!
    dummy_mask[:, 15] = 0  # 모든 배치에서 '15번 위치'에 비트를 두는 것을 물리적으로 금지!

    # 4. 순전파 (Forward Pass) 실행 - 뇌를 통과시킴
    print("\n[테스트 시작] QECNet에 텐서를 통과시킵니다...")
    policy_probs, value = net(dummy_state, dummy_mask)

    # --- 엄격한 검증 (Assertions) ---

    # 검증 1: 텐서 차원(Shape)이 우리가 설계한 대로 정확하게 나오는가?
    assert policy_probs.shape == (batch_size, action_size), f"정책망 형태 오류: {policy_probs.shape}"
    assert value.shape == (batch_size, 1), f"가치망 형태 오류: {value.shape}"

    # 검증 2: Softmax가 제대로 적용되어 확률의 총합이 1(100%)이 되는가?
    sums = torch.sum(policy_probs, dim=1)
    assert torch.allclose(sums, torch.ones(batch_size)), "확률의 합이 1이 아닙니다!"

    # 검증 3: 마스킹(Masking) 기능이 완벽하게 작동하는가? (가장 중요)
    # 금지시킨 0번과 15번 행동의 확률이 정확히 0.0으로 떨어졌는지 확인합니다.
    assert policy_probs[0, 0].item() == 0.0, "치명적 오류: 마스킹된 0번 행동의 확률이 0이 아닙니다!"
    assert policy_probs[0, 15].item() == 0.0, "치명적 오류: 마스킹된 15번 행동의 확률이 0이 아닙니다!"

    # 검증 4: Value(가치) 값이 tanh 함수를 거쳐 -1(최악) ~ 1(최상) 사이에 안정적으로 존재하는가?
    assert torch.all((value >= -1.0) & (value <= 1.0)), "가치망 출력이 -1과 1 사이를 벗어났습니다."

    print("✅ QECNet 텐서 차원 및 마스킹 검증을 완벽하게 통과했습니다!")

if __name__ == "__main__":
    test_qecnet_forward_pass()