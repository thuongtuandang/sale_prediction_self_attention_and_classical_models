import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

class NormalizedLayer(nn.Module):
    def __init__(self, input_size, input_length):
        super().__init__()
        # input_size is the dimension of input
        self.input_size = input_size
        # N is the batch size
        N = input_length
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(normalized_shape=(N,self.input_size))
    
    def forward(self, inputs, self_attention_output):
        input_norm = inputs + self_attention_output
        # Note that the input for nn.Layernorm is of shape [*, batch_length, input_size]
        # so we need to reshape the input_norm to shape [batch_length, 1, input_size]
        # input_norm = input_norm.view(-1, 1, self.input_size)
        normalized_input_norm = self.norm(input_norm)
        output = self.dropout(normalized_input_norm)
        return output