import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from bitnet_linear_158 import BitLinear158

def scaled_dot_product_attention(query, key, value, dropout=0.0):
    scale = query.size(-1) ** 0.5
    similarity = torch.matmul(query / scale, key.transpose(-2, -1))
    attention_weights = F.softmax(similarity, dim=-1)
    if dropout > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout)
    output = torch.matmul(attention_weights, value)
    return output


class BitnetMultiAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        query_heads: int = 8,
        kv_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
    ):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.gamma_init = gamma_init

        self.q_proj = BitLinear158(embed_dim, embed_dim, bias=bias)
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = BitLinear158(embed_dim, kv_embed_dim, bias=bias)
        self.v_proj = BitLinear158(embed_dim, kv_embed_dim, bias=bias)
        self.norm = nn.LayerNorm(kv_embed_dim, eps=layer_norm_eps)
        self.out_proj = BitLinear158(kv_embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)
        
        x = scaled_dot_product_attention(q, k, v, dropout=self.dropout)
        x = rearrange(x, "b n h d -> b n (h d)")

        x = self.norm(x)
        x = self.out_proj(x)

        return x