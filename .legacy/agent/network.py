import torch
import torch.nn as nn
import torch.nn.functional as F

class QECNet(nn.Module):
    def __init__(self, num_qubits, num_stabilizers):
        super(QECNet, self).__init__()
        
        self.n = num_qubits
        self.m = num_stabilizers
        self.action_size = 2 * self.m * self.n  # 총 가능한 행동의 수
        
        # --- 공유된 특징 추출기 (Shared Backbone) ---
        # (2, m, n) 형태의 패리티 행렬을 이미지처럼 처리합니다.
        # 양자 코드의 특성상 m과 n이 작으므로 커널 사이즈를 작게 유지합니다.
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Conv 레이어를 통과한 후 Flatten 했을 때의 차원 계산
        self.flatten_size = 128 * self.m * self.n
        
        # --- 1. 정책 헤드 (Policy Head) ---
        # "다음에 어디에 비트를 추가할까?" (확률 분포)
        self.policy_conv = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * self.m * self.n, self.action_size)
        
        # --- 2. 가치 헤드 (Value Head) ---
        # "이 구조의 최종 논리적 에러율은 낮을까?" (상태 평가 점수 -1 ~ 1)
        self.value_conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * self.m * self.n, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x, action_mask=None):
        """
        x: [batch_size, 2, m, n] 형태의 상태 텐서
        action_mask: 교환 법칙을 위반하는 행동을 0으로 강제하는 마스크
        """
        # Backbone 연산 (CNN으로 공간적 패턴 추출)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # --- Policy Head 연산 ---
        p = F.relu(self.policy_conv(x))
        p = p.view(-1, 2 * self.m * self.n)  # Flatten
        p = self.policy_fc(p)
        
        # 마스킹(Masking) 적용: 불가능한 행동은 확률이 계산되지 않도록 -무한대로 설정
        if action_mask is not None:
            # action_mask가 0인 곳은 -1e9(매우 작은 수)로 만들어 Softmax를 통과하면 0%가 되게 함
            p = p.masked_fill(action_mask == 0, -1e9)
            
        policy_probs = F.softmax(p, dim=1)
        
        # --- Value Head 연산 ---
        v = F.relu(self.value_conv(x))
        v = v.view(-1, 1 * self.m * self.n)  # Flatten
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # -1(매우 나쁨) ~ 1(매우 좋음) 사이의 점수 출력
        
        return policy_probs, value