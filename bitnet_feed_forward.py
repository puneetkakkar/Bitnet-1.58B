from typing import Optional
from torch import nn
from bitnet_linear_158 import BitLinear158

class BitnetFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        post_act_ln: bool = False,
        no_bias: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.ff = nn.Sequential(
            BitLinear158(dim, inner_dim, bias=not no_bias), 
            nn.GELU(),
            nn.LayerNorm(inner_dim) if post_act_ln else None,
            BitLinear158(inner_dim, dim_out, bias=not no_bias),
        )

    def forward(self, x):
        return self.ff(x)
