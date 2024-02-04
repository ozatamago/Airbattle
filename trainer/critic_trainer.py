import torch
import torch.optim as optim
import torch.nn.functional as F

class ValueFunctionTrainer:
    def __init__(self, value_network, learning_rate=1e-3, gamma=0.99):
        self.value_network = value_network
        self.optimizer = optim.Adam(value_network.parameters(), lr=learning_rate)
        self.gamma = gamma

    def calculate_target(self, rewards, next_states, dones, critic_network, gamma=0.99, lambda_gae=0.95):
        y_lambda = torch.zeros_like(rewards)
        next_value = critic_network(next_states[-1]).detach()
        for t in reversed(range(len(rewards))):
            next_value = rewards[t] + gamma * next_value * (1 - dones[t])
            y_lambda[t] = next_value
            for n in range(1, len(rewards) - t):
                G_tn = sum([gamma ** (l - 1) * rewards[t + l - 1] for l in range(1, n + 1)])
                if t + n < len(rewards):
                    G_tn += gamma ** n * critic_network(next_states[t + n]).detach()
                y_lambda[t] += (1 - lambda_gae) * lambda_gae ** (n - 1) * G_tn
        return y_lambda

    def train_step(self, states, rewards, next_states, dones):
        # 価値関数 Vϕ の計算
        predicted_values = self.value_network(states)

        # 目標 y(λ) の計算
        target_values = self.calculate_target(rewards, next_states, dones, self.gamma)

        # 損失 J(ϕ) の計算
        loss = F.mse_loss(predicted_values, target_values)

        # 価値ネットワークの更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# 学習ループの例
# # 価値関数ネットワークのインスタンス化
# value_network = ...  # 価値関数ネットワークのインスタンス化

# # トレーナークラスのインスタンス化
# trainer = ValueFunctionTrainer(value_network)

# # 学習データのロード
# states = ...  # 状態
# rewards = ...  # 報酬
# next_states = ...  # 次の状態
# dones = ...  # エピソード終了フラグ

# # 学習ステップの実行
# loss = trainer.train_step(states, rewards, next_states, dones)