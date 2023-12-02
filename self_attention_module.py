import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention
from normalized_layer import NormalizedLayer

class SelfAttentionModule(nn.Module):
    def __init__(self, input_size, heads, mask = True):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.pos_encoding = PositionalEncoding(self.input_size)
        self.multi_head_attention = MultiHeadAttention(self.input_size, heads, mask)
        self.norm = NormalizedLayer(self.input_size)
        self.linear = nn.Linear(self.input_size, 1)
        self.mask = mask

        # Define 
        # self.multi_head_attention_params = nn.Parameter(self.multi_head_attention.parameters())
        # self.norm_params = nn.Parameter(self.norm.parameters())
        # self.linear_params = nn.Parameter(self.linear.parameters())

    
    def forward(self, input):
        pos = self.pos_encoding.forward(input)
        att = self.multi_head_attention.forward(pos)
        norm_att = self.norm.forward(pos, att)
        output = self.linear(norm_att)
        return output
    
    def process(self, X, y, run_backward = False):
        MSELoss = 0
        for x, y_true in zip(X,y):
            x = torch.tensor(x).float().unsqueeze(0)
            y_true = torch.tensor(y_true).float()
            y_pred = self.forward(x).squeeze()
            loss = self.criterion(y_pred, y_true)
            MSELoss += loss.item()
            if run_backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return MSELoss

    def fit(self, X, y, num_epochs = 200, learning_rate = 0.01, print_period = 20):
        # Define loss and optimizer
        self.criterion = nn.MSELoss()
        self.parameters = [
            *self.multi_head_attention.parameters(),
            *self.norm.parameters(),
            *self.linear.parameters()
        ]
        self.optimizer = optim.Adam(self.parameters, lr=learning_rate)
        for i in range(num_epochs):
            MSELoss = self.process(X, y, run_backward=True)
            RMSELoss = np.sqrt(MSELoss/X.shape[0])
            if i%print_period == 0:
                print(f'Step: {i}')
                print(f"RMSE loss: {RMSELoss}")
    
    def predict(self, X, y):
        return self.process(X, y, run_backward=False)
