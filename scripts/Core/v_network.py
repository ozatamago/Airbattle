import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer import TransformerEncoder as t


class VNetwork(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(VNetwork, self).__init__()
        self.transformer_encoder = t(num_layers, d_model, num_heads, d_ff, dropout)
        self.value_head = nn.Linear(d_model, 1)  # 状態の価値を計算するヘッド

    def forward(self, observations):
        # observations: バッチサイズ x シーケンス長 x d_model
        encoded = self.transformer_encoder(observations)  # Transformerエンコーダによる観測のエンコード
        value = self.value_head(encoded.mean(dim=1))  # エンコーディングされた観測の平均をとり、価値を計算
        return value

# 使用例
# # ハイパーパラメータの設定
# d_model = 512
# num_heads = 8
# d_ff = 2048
# num_layers = 3
# dropout = 0.1

# # クリティックモデルのインスタンス化
# critic = CriticModel(d_model, num_heads, d_ff, num_layers, dropout)

# # サンプル観測データ
# observations = torch.rand(10, 32, d_model)  # バッチサイズ x シーケンス長 x d_model

# # フォワードパス
# state_values = critic(observations)