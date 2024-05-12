import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn
from bitnet_linear_158 import BitLinear158

class BitnetMultiGroupQueryAttention(nn.Module):
    def __init__( self, embed_dim, query_heads=8, kv_heads=4, dropout=0.1, bias=True):
        super().__init__()
        
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout

        layer_norm_eps = 1e-5

        # Initialize the linear projections
        self.q_proj = BitLinear158(embed_dim, embed_dim, bias=bias)
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = BitLinear158(embed_dim, kv_embed_dim, bias=bias)
        self.v_proj = BitLinear158(embed_dim, kv_embed_dim, bias=bias)

        # Initialize normalization layer
        self.norm = nn.LayerNorm( kv_embed_dim, eps=layer_norm_eps)

        # Initialize output projection
        self.out_proj = BitLinear158(kv_embed_dim, embed_dim, bias=bias)

        # Resetting the parameters
        self._reset_parameters()

    def scaled_dot_product_attention(self, query, key, value):
    
        query = rearrange(query, "b n h d -> b h n d")
        key = rearrange(key, "b s h d -> b h s d")
        value = rearrange(value, "b s h d -> b h s d")

        bq, hq, nq, _ = query.shape
        _, hk, nk, _ = key.shape

        # Here, we are scaling the factor calculation
        scale = query.size(-1) ** 0.5
        query = query / scale

        num_head_groups = hq // hk
        if num_head_groups > 1:
            query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
            similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
        else:
            similarity = einsum(query, key, "b h n d, b h s d -> b h n s")

        mask = torch.ones((bq, nq, nk), device=query.device, dtype=torch.bool).tril_()
        mask_dims = (mask.ndim == 2, mask.ndim == 3)

        if mask_dims == (True, False):
            mask = rearrange(mask, "b s -> b () () s")
        elif mask_dims == (False, True):
            mask = rearrange(mask, "b n s -> b () n s")
            
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

        attention = F.softmax(similarity / scale, dim=-1)
        
        attention = F.dropout(attention, p=self.dropout)

        out = einsum(attention, value, "b h n s, b h s d -> b h n d")
        out = rearrange(out, "b h n d -> b n h d")

        return out

    def _reset_parameters(self):
        # Resetting the initialized parameters
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward( self, query, key, value ):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Re-arranging the dimensions for multi-head attention
        q = q.view(q.size(0), -1, self.query_heads, q.size(-1) // self.query_heads)
        k = k.view(k.size(0), -1, self.kv_heads, k.size(-1) // self.kv_heads)
        v = v.view(v.size(0), -1, self.kv_heads, v.size(-1) // self.kv_heads)
        
        # Scaled dot-product attention
        x = self.scaled_dot_product_attention(q, k, v)

        # Re-arranging dimensions and then applying layer normalization if it is set to true
        x = x.reshape(x.size(0), -1, x.size(-1) * self.kv_heads)

        if self.norm:
            x = self.norm(x)

        x = self.out_proj(x)

        return x