import torch
from typing import Optional
from torch import nn
from bitnet_linear_158 import BitLinear158

def default(val, d):
    return val if val is not None else d

def init_zero_(tensor):
    nn.init.constant_(tensor, 0.0)

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation,
        mult_bias = False,
        linear = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        self.mult_bias = mult_bias

        if linear:
            self.proj = linear(dim_in, dim_out * 2)
        else:
            self.proj = BitLinear158(dim_in, dim_out * 4, *args, **kwargs)

        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate) * self.mult_bias

class BitnetFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out: Optional[int] = None,
        mult = 4,
        glu = False,
        glu_mult_bias = False,
        swish = False,
        post_act_ln = False,
        dropout = 0.0,
        no_bias = False,
        zero_init_output = False,
        *args,
        **kwargs
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias=glu_mult_bias)
        else:
            project_in = nn.Sequential(
                BitLinear158(dim, inner_dim, bias=not no_bias, *args, **kwargs), activation
            )

        self.ff = nn.Sequential(
            project_in,
            nn.GELU(),
            nn.LayerNorm(inner_dim) if post_act_ln else None,
            BitLinear158(inner_dim, dim_out, bias=not no_bias),
        )

        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)
