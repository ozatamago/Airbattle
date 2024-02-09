import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)
        return output, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.linear_keys = nn.Linear(d_model, d_model)
        self.linear_values = nn.Linear(d_model, d_model)
        self.linear_queries = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        def transform(x, linear):
            return linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        query, key, value = [transform(x, linear) for x, linear in zip((query, key, value), (self.linear_queries, self.linear_keys, self.linear_values))]

        x, _ = self.attention(query, key, value, mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.output_linear(x)

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src = src + self.self_attn(src2, src2, src2, src_mask)
        src2 = self.norm2(src)
        src = src + self.dropout(self.feed_forward(src2))
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


# 使用例
# # ハイパーパラメータの設定
# num_layers = 3
# d_model = 512
# num_heads = 8
# d_ff = 2048
# dropout = 0.1

# # エンコーダのインスタンス化
# encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)

# # サンプル入力データ
# src = torch.rand(10, 32, d_model)  # (batch_size, seq_len, d_model)

# # フォワードパス
# output = encoder(src)
