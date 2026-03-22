import torch
import torch.nn as nn
import torch.nn.functional as F

class QEC_GFlowNet(nn.Module):
    def __init__(self, num_qubits=9, num_stabilizers=4):
        super(QEC_GFlowNet, self).__init__()
        self.n = num_qubits
        self.m = num_stabilizers
        
        # 🌟 1. GFlowNet의 심장: log Z (파티션 함수)
        # 전체 탐색 공간의 총 보상합을 추정하는 단일 스칼라 값입니다.
        # 신경망의 입력과 상관없이 독립적으로 학습되는 아주 특별한 파라미터입니다!
        self.logZ = nn.Parameter(torch.tensor([0.0]))
        
        # 2. 공통 특성 추출기 (Feature Extractor)
        # 기존 CNN이나 MLP 구조를 그대로 가져와도 무방합니다.
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc_input_dim = 128 * self.m * self.n
        
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        
        # 🌟 3. Forward Policy (P_F) 출력층
        # Value(승률)를 출력하는 층은 완전히 삭제되었습니다!
        # 오직 "어느 위치에 선을 그을 것인가?" 에 대한 로짓(Logit)만 출력합니다.
        action_space_size = (2 * self.m * self.n) + 1
        self.policy_head = nn.Linear(256, action_space_size)

    def forward(self, state):
        """
        Args:
            state: (batch_size, 2, m, n) 형태의 양자 회로 상태 텐서
        Returns:
            P_F_logits: Forward 확률을 계산하기 위한 로짓 값
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.fc_input_dim) # Flatten
        
        x = F.relu(self.fc1(x))
        
        # Softmax는 Loss 계산할 때(또는 샘플링할 때) 적용할 것이므로 Logit 상태로 반환합니다.
        P_F_logits = self.policy_head(x) 
        
        return P_F_logits