import math
import numpy as np
import torch

class Node:
    """MCTS 트리의 각 상태(State)를 나타내는 노드"""
    def __init__(self, state, parent=None, action_taken=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob

    @property
    def q_value(self):
        """평균 가치 (Q = W / N)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        """자식 노드가 하나라도 있으면 확장된 것으로 간주"""
        return len(self.children) > 0


class MCTS:
    """AlphaZero 스타일의 몬테카를로 트리 탐색 에이전트"""
    def __init__(self, network, env, num_simulations=100, c_puct=1.5):
        self.network = network
        self.env = env
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    @torch.no_grad()
    def search(self, initial_state):
        root = Node(state=initial_state)
        
        # 🌟 핵심 1: 신경망(QECNet)이 현재 어디(CPU인지 GPU인지)에 올라가 있는지 그 주소를 가져옵니다.
        device = next(self.network.parameters()).device
        
        for _ in range(self.num_simulations):
            node = root
            
            # --- 1. 선택 (Selection) ---
            while node.is_expanded():
                node = self._select_child(node)
                
            # --- 2. 확장 및 평가 (Expansion & Evaluation) ---
            self.env.state = node.state.copy()
            valid_mask = self.env.get_valid_action_mask()
            
            # 더 이상 둘 곳이 없는 막다른 길(Dead End) 처리
            if np.sum(valid_mask) == 0:
                self._backpropagate(node, -1.0)
                continue
                
            # 🌟 핵심 2: 상태와 마스크를 텐서로 변환함과 동시에 `.to(device)`를 붙여 GPU로 쏘아 올립니다!
            # 여기서 단 하나라도 누락되면 Expected all tensors to be on the same device 에러가 터집니다.
            state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0).to(device)
            mask_tensor = torch.tensor(valid_mask, dtype=torch.float32).unsqueeze(0).to(device)
            
            # GPU 위에서 신경망 연산 수행
            policy_probs, value = self.network(state_tensor, mask_tensor)
            
            # 🌟 핵심 3: 연산이 끝난 결과를 다시 CPU 메모리의 Numpy 배열로 안전하게 내려받습니다.
            policy_probs = policy_probs.squeeze(0).cpu().numpy()
            value = value.item()
            
            # 유효한 행동들에 대해 자식 노드 확장
            for action, prob in enumerate(policy_probs):
                if valid_mask[action] == 1 and prob > 0:
                    next_state = node.state.copy()
                    c = action // (self.env.m * self.env.n)
                    rem = action % (self.env.m * self.env.n)
                    r, col = rem // self.env.n, rem % self.env.n
                    next_state[c, r, col] = 1
                    
                    node.children[action] = Node(state=next_state, parent=node, 
                                                 action_taken=action, prior_prob=prob)
            
            # --- 3. 역전파 (Backpropagation) ---
            self._backpropagate(node, value)

        action_probs = np.zeros(self.env.action_space.n)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
            
        action_probs /= np.sum(action_probs)
        return action_probs

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        
        for action, child in node.children.items():
            q_val = child.q_value
            u_val = self.c_puct * child.prior_prob * (math.sqrt(node.visit_count) / (1 + child.visit_count))
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def _backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value 
            node = node.parent