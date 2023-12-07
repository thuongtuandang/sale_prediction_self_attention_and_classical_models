import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, input_size, input_length):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pos_encoding = torch.empty(0, input_size)
        iter = input_length
        for k in range(iter):
            d = input_size
            pos = torch.zeros((1,d))
            for i in range(d):
                if i % 2 == 0:
                    pos[0][i] = np.sin((k/(10000**(i/d))))
                if i % 2 == 1:
                    pos[0][i] = np.cos((k/(10000**((i-1)/d))))
            pos_encoding = torch.cat((pos_encoding, pos), dim = 0)
        self.pe = pos_encoding
    

    # def __init__(self, input_size, input_length):
    #     super().__init__()
    #     self.dropout = nn.Dropout(p=0.1)
    #     iter = input_length
    #     d = input_size
    #     position = torch.arange(iter).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    #     pe = torch.zeros(iter, d)
    #     pe[:, 0::2] = torch.sin(position * div_term)
    #     pe[:, 1::2] = torch.cos(position * div_term)
    #     self.pe = pe
    
    def forward(self, inputs):
        output = inputs + self.pe
        return self.dropout(output)

    
