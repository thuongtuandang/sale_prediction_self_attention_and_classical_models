import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

class NormalizedLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, input, self_attention_output):
        input_norm = input + self_attention_output
        self.norm = nn.LayerNorm(normalized_shape=(1,self.input_size))
        normalized_input_norm = self.norm(input_norm)
        output = self.dropout(self.norm(normalized_input_norm))
        return output