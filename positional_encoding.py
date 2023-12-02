import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # def __init__(self, input_size):
    #     super().__init__()
    #     self.dropout = nn.Dropout(p = 0.1)
    #     self.pe = torch.zeros(1, input_size)
    #     # This create a vector of shape (1, input_size)
    #     # And elements from 1 to input_size
    #     position = torch.arange(0, input_size, dtype = torch.float).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, input_size, 2).float() * (-math.log(10000.0)/input_size))
    #     print(f"position shape: {position.shape} ")
    #     print(f"div_team.shape: {div_term.shape}")
    #     self.pe[0, 0::2] = torch.sin(position * div_term)
    #     self.pe[0, 1::2] = torch.cos(position * div_term)
    
    # Positional encoding for the whole sentence
    def __init__(self, input_size):
        super().__init__()
        d = input_size
        position = torch.arange(1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(1, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, input):
        x = input + self.pe
        return self.dropout(x)

    
