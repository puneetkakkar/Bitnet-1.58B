import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from bitnet_feed_forward import BitnetFeedForward
from bitnet_multi_group_query_attention import BitnetMultiGroupQueryAttention


class RMSNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale_factor = dim ** 0.5
        self.affine = nn.Parameter(torch.ones(dim)) if affine else 1.0

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.affine * self.scale_factor


class TransformerLayer(nn.Module):

    def __init__( self, dim, heads, ff_mult=2, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        self.multi_head_attn = BitnetMultiGroupQueryAttention(dim, heads, dropout=attn_dropout)

        self.feed_forward = BitnetFeedForward(dim, dim, ff_mult)

    def forward(self, x):
        skip_connection = x
        x = self.multi_head_attn(x, x, x)
        x = x + skip_connection
        x = self.feed_forward(x) + x
        return x


class BitNetTransformer(nn.Module):

    def __init__( self, token_dim, depth, num_tokens, num_heads=8, ff_mult=4):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, token_dim)
        self.transformer = nn.ModuleList([
            TransformerLayer(token_dim, num_heads, ff_mult=ff_mult) for _ in range(depth)
        ])
        self.token_projection = nn.Sequential(RMSNorm(token_dim), nn.Linear(token_dim, num_tokens))

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.transformer:
            x = layer(x)
        return self.token_projection(x)
