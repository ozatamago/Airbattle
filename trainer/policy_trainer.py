import torch
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetworkTrainer:
    def __init__(self, policy_network, critic_network, q_network, learning_rate=1e-3, gamma=0.99, lambda_gae=0.95):
        self.policy_network = policy_network
        self.critic_network = critic_network
        self.q_network = q_network
        self.optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.lambda_gae = lambda_gae

    @staticmethod
    def calculate_advantage(rewards, states, next_states, dones, critic_network, q_network, gamma=0.99, lambda_gae=0.95):
        # TD(λ)によるy(λ)の計算
        y_lambda = 0
        for n in range(1, len(rewards) + 1):
            G_tn = sum([gamma ** (l - 1) * rewards[l - 1] for l in range(1, n + 1)])
            if n < len(rewards):
                G_tn += gamma ** n * critic_network(next_states[n]).item()
            y_lambda += (1 - lambda_gae) * lambda_gae ** (n - 1) * G_tn

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
