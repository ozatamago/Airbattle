import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

class PolicyNetworkTrainer:
    def __init__(self, policy_network, critic_network, q_network, learning_rate=1e-3, gamma=0.99, lambda_gae=0.95):
        self.policy_network = policy_network
        self.critic_network = critic_network
        self.q_network = q_network
        self.optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.lambda_gae = lambda_gae

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

    # @staticmethod
    def calculate_advantage(self, rewards, states, next_states, dones, critic_network, q_network, gamma=0.99, lambda_gae=0.95):
        # TD(λ)によるy(λ)の計算
        y_lambda = self.calculate_target(rewards, next_states, dones, gamma)

        # 状態行動価値関数 Qψ の計算
        Q_values = q_network(states)

        # Advantageの計算
        advantages = y_lambda - Q_values
        return advantages

    def train_step(self, states, actions, rewards, next_states, dones):
        # 方策勾配の計算
        advantages = self.calculate_advantage(rewards, states, next_states, dones, self.critic_network, self.q_network, self.gamma, self.lambda_gae)
        log_probs = self.policy_network(states).log_prob(actions)
        policy_loss = -(log_probs * advantages).mean()

        # 方策ネットワークの更新
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()
    

# 学習ループの例
# # 方策ネットワークとクリティックネットワーク、およびQネットワークのインスタンス化
# policy_network = ...  # PolicyNetworkのインスタンス
# critic_network = ...  # CriticNetworkのインスタンス
# q_network = ...  # QNetworkのインスタンス

# # 学習用トレーナーのインスタンス化
# trainer = PolicyNetworkTrainer(policy_network, critic_network, q_network)

# # 学習パラメータ
# num_episodes = 1000  # 総エピソード数
# max_steps = 100  # 各エピソードの最大ステップ数

# for episode in range(num_episodes):
#     states = []  # 状態を保存するリスト
#     actions = []  # 行動を保存するリスト
#     rewards = []  # 報酬を保存するリスト
#     next_states = []  # 次の状態を保存するリスト
#     dones = []  # エピソード終了フラグを保存するリスト

#     # 環境の初期化
#     state = env.reset()
#     for step in range(max_steps):
#         # 方策ネットワークから行動を選択
#         action = policy_network.select_action(state)
        
#         # 選択した行動を環境に適用
#         next_state, reward, done, _ = env.step(action)
        
#         # データの保存
#         states.append(state)
#         actions.append(action)
#         rewards.append(reward)
#         next_states.append(next_state)
#         dones.append(done)

#         # 次の状態への更新
#         state = next_state

#         if done:
#             break

#     # 1エピソード分の学習
#     policy_loss = trainer.train_step(states, actions, rewards, next_states, dones)
#     print(f"Episode {episode}, Loss: {policy_loss}")

#     # エピソード終了処理（任意で追加可能）
#     ...

