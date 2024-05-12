import torch
from typing import Optional
from torch import nn
from bitnet_linear_158 import BitLinear158

def default(val, d):
    return val if val is not None else d

def init_zero_(tensor):
    nn.init.constant_(tensor, 0.0)

class BitnetFeedForward(nn.Module):
    def __init__( self, input_dim, output_dim = None, multiplier = 4, post_activation_layer_norm = True, no_bias = False):
        super().__init__()
        inner_dim = int(input_dim * multiplier)
        output_dim = default(output_dim, input_dim)

        project_in = nn.Sequential(BitLinear158(input_dim, inner_dim, bias=not no_bias), nn.SiLU())

        layers = [
            project_in,
            nn.GELU(),
            nn.LayerNorm(inner_dim) if post_activation_layer_norm else None,
            BitLinear158(inner_dim, output_dim, bias=not no_bias),
        ]

        self.ff = nn.Sequential(*[layer for layer in layers if layer])

    def forward(self, x):
        return self.ff(x)
