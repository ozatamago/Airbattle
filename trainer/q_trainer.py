import torch
import torch.optim as optim
import torch.nn.functional as F
from models.v_network import VNetwork as v
from models.q_network import QNetwork as q

class QTrainer:
    def __init__(self, q, v, learning_rate=1e-3, gamma=0.99, lambda_gae=0.95):
        self.q = q
        self.v = v
        self.optimizer = optim.Adam(q.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.lambda_gae = lambda_gae

    def calculate_target(self, rewards, next_states, dones):
        y_lambda = torch.zeros_like(rewards)
        next_value = self.v(next_states[-1]).detach()
        for t in reversed(range(len(rewards))):
            next_value = rewards[t] + self.gamma * next_value * (1 - dones[t])
            y_lambda[t] = next_value
            for n in range(1, len(rewards) - t):
                G_tn = sum([self.gamma ** (l - 1) * rewards[t + l - 1] for l in range(1, n + 1)])
                if t + n < len(rewards):
                    G_tn += self.gamma ** n * self.v(next_states[t + n]).detach()
                y_lambda[t] += (1 - self.lambda_gae) * self.lambda_gae ** (n - 1) * G_tn
        return y_lambda

    def train_step(self, states, actions, rewards, next_states, dones):
        # Q値の計算
        predicted_q_values = self.q(states, actions)

        # 目標 Q 値の計算
        target_q_values = self.calculate_target(rewards, next_states, dones)

        # 損失の計算
        loss = F.mse_loss(predicted_q_values, target_q_values)

        # Qネットワークの更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# 使用例
# # Qネットワークとクリティックネットワークのインスタンス化
# q_network = ...  # Qネットワークのインスタンス化
# critic_network = ...  # クリティックネットワークのインスタンス化

# # トレーナーのインスタンス化
# trainer = QTrainer(q_network, critic_network)

# # 学習データのロード
# states = ...  # 状態
# actions = ...  # 行動
# rewards = ...  # 報酬
# next_states = ...  # 次の状態
# dones = ...  # エピソード終了フラグ

# # 学習ステップの実行
# loss = trainer.train_step(states, actions, rewards, next_states, dones)
