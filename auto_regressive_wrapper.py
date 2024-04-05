import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

class AutoRegressiveWrapper(nn.Module):

    def __init__(self, net, max_sequence_length=2048, pad_value=0):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.pad_value = pad_value
        self.net = net

    def generate(self, start_tokens, sequence_length, eos_token=None, temperature=1.0, filter_thres=0.9, **kwargs):
        b, t, device = *start_tokens.shape, start_tokens.device
        out = start_tokens

        for _ in range(sequence_length):
            logits = self.net(out, **kwargs)[:, -1, :]
            filtered_logits = self.top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)

            if eos_token is not None and (out == eos_token).any(dim=-1).all():
                shifted_is_eos_tokens = F.pad(out == eos_token, (1, -1))
                mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                out = out.masked_fill(mask, self.pad_value)
                break

        out = out[:, t:]
        return out

    def top_k(self, logits, thres=0.9):
        k = int((1 - thres) * logits.shape[-1])
        val, ind = torch.topk(logits, k)
        probs = torch.full_like(logits, float("-inf"))
        probs.scatter_(1, ind, val)
        return probs

    def forward(self, x, **kwargs):
        x_inp, x_labels = x[:, :-1], x[:, 1:]
        logits = self.net(x_inp, **kwargs)
        return F.cross_entropy(rearrange(logits, "b c n -> b n c"), x_labels)