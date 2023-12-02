import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # Positional encoding for the whole batch
    def __init__(self, input_size, batch_length):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        iter = batch_length
        d = input_size
        position = torch.arange(iter).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(iter, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe
    
    def forward(self, inputs):
        output = inputs + self.pe
        return self.dropout(output)

    
