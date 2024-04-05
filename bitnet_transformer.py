import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from bitnet_feed_forward import BitnetFeedForward
from bitnet_multi_head_attention import BitnetMultiAttention


class RMSNorm(nn.Module):

    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.0

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class Transformer(nn.Module):

    def __init__(
        self, dim: int, heads: int, depth: int, ff_mult: int = 2, *args, **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(BitnetMultiAttention(dim, heads, *args, **kwargs))

            self.ffn_layers.append(
                BitnetFeedForward(
                    dim,
                    dim,
                    ff_mult,
                    post_act_ln=True,
                    # dropout=0.1,
                ),
            )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        skip = x
        for attn, ffn in zip(self.layers, self.ffn_layers):
            x = attn(x, x, x, *args, **kwargs)
            x = x + skip
            x = ffn(x) + x
        return x


class BitNetTransformer(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        num_tokens: int,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)
        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, ff_mult=ff_mult
        )
        self.to_logits = nn.Sequential(
                            RMSNorm(dim), 
                            nn.Linear(dim, num_tokens)
                        )

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)
