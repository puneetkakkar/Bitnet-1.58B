import torch
import torch.nn.functional as F
from torch import nn

class AutoRegressiveWrapper(nn.Module):

    def __init__(self, model, max_sequence_length=1048, padding_value=0):
        super().__init__()
        """
            Here we initialize the AutoRegressiveWrapper
        """
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value
        self.model = model

    """
        This function generates the text sequences autoregressively.
        It iteratively predicts the next token in the sequence using 
        the provided model. Now, this process continues for a specified sequence 
        length or until an end-of-sequence token is encountered. To generate the 
        sequences it incorporates techniques such as top-k filtering and temperature 
        scaling to control the diversity and quality of the generated sequences. 
    """
    @torch.no_grad()
    def generate(self, start_tokens, sequence_length, eos_token=None, temperature=1.0, filter_threshold=0.9, **kwargs):    
        out = start_tokens

        for _ in range(sequence_length):
            logits = self.model(out, **kwargs)[:, -1, :]
            filtered_logits = self.filter_top_k(logits, threshold=filter_threshold)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)

            if eos_token is not None and (out == eos_token).any(dim=-1).all():
                eos_mask = (out == eos_token).float().cumsum(dim=-1) >= 1
                out = out.masked_fill(F.pad(eos_mask, (1, -1)), self.padding_value)
                break

        out = out[:, start_tokens.shape[1]:]
        return out

    """
        This function applies top-k filtering over the sequences logits. This function 
        selects the top-k values from a tensor of logits, setting all other values to 
        negative infinity. 
    """
    def filter_top_k(self, logits, threshold=0.9):
        k = int((1 - threshold) * logits.shape[-1])
        top_values, top_indices = torch.topk(logits, k)

        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(1, top_indices, top_values)

        return filtered_logits

    """
        This function defines the forward pass of the autoregressive wrapper model.
        It takes an input sequence, and computes the loss by comparing the model's predictions 
        with the ground truth labels.
    """
    def forward(self, x, **kwargs):
        input_sequence, target_sequence = x[:, :-1], x[:, 1:]
        logits = self.model(input_sequence, **kwargs)
        loss = F.cross_entropy(logits.transpose(1, 2), target_sequence)
        
        return loss