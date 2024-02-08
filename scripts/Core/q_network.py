import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer as t

class QNetwork(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, d_observation, d_action, d_embedding, dropout=0.1):
        super(QNetwork, self).__init__()
        self.observation_encoder = nn.Linear(d_observation, d_embedding)  # gj: エージェントjの観測エンコーダ
        self.obs_action_encoder = nn.Linear(d_observation + d_action, d_embedding)  # fi: 他のエージェントの観測行動ペアエンコーダ
        self.transformer_encoder = t.TransformerEncoder(num_layers, d_embedding, num_heads, d_ff, dropout)
        self.value_head = nn.Linear(d_embedding, 1)  # 状態行動価値を計算するヘッド

    def forward(self, observation_j, observations_actions_others):
        # observation_j: エージェントjの観測
        # observations_actions_others: 他のエージェントの観測行動ペア
        encoded_obs_j = self.observation_encoder(observation_j)
        encoded_obs_act_others = torch.cat([self.obs_action_encoder(obs_act) for obs_act in observations_actions_others], dim=0)
        encoded = torch.cat([encoded_obs_j.unsqueeze(0), encoded_obs_act_others.unsqueeze(0)], dim=1)
        output = self.transformer_encoder(encoded)
        q_value = self.value_head(output.mean(dim=1))
        return q_value


# 使用例
# # ハイパーパラメータの設定
# d_model = 512
# num_heads = 8
# d_ff = 2048
# num_layers = 3
# d_observation = 256
# d_action = 128
# d_embedding = 512
# dropout = 0.1

# # Qネットワークのインスタンス化
# q_network = QNetwork(d_model, num_heads, d_ff, num_layers, d_observation, d_action, d_embedding, dropout)

# # サンプルデータ
# observation_j = torch.rand(d_observation)  # エージェントjの観測
# observations_actions_others = [torch.rand(d_observation + d_action) for _ in range(4)]  # 他のエージェントの観測行動ペア

# # フォワードパス
# q_value = q_network(observation_j, observations_actions_others)
