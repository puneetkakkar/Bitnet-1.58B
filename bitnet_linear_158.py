from torch import nn, Tensor
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm
import torch.nn.functional as F

"""
    Quantizes the activations using symmetric quantization.
"""
def activation_quant(x: Tensor):
    activation_scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    quantized_x = (x * activation_scale).round().clamp_(-128, 127) / activation_scale
    return quantized_x

"""
    Quantizes the weights using mean-based symmetric quantization.
"""
def quantize_weight(w: Tensor):
    weight_scale = w.abs().mean()
    weight_mean = w.mean()
    quantized_weight = (w - weight_mean).sign() * weight_scale
    return quantized_weight

class BitLinear158(nn.Linear):
    
    """
        Performs a forward pass through the BitLinear layer.
    """
    def forward(self, x: Tensor) -> Tensor:
        normalized_x = SimpleRMSNorm(self.in_features)(x)
        quantized_x = normalized_x + (activation_quant(normalized_x) - normalized_x).detach()
        quantized_weight = self.weight + (quantize_weight(self.weight) - self.weight).detach()
        y = F.linear(quantized_x, quantized_weight, self.bias)
        return y
