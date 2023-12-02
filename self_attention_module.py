import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention
from normalized_layer import NormalizedLayer

class SelfAttentionModule(nn.Module):
    def __init__(self, heads, input_size, batch_length = 32, mask = True):
        super().__init__()
        # input_size is the dimension of inputs
        self.input_size = input_size
        self.batch_length = batch_length
        self.heads = heads
        self.pos_encoding = PositionalEncoding(self.input_size, self.batch_length)
        self.multi_head_attention = MultiHeadAttention(input_size, batch_length, heads, mask)
        self.norm = NormalizedLayer(self.input_size, batch_length)
        self.linear = nn.Linear(self.input_size, 1)
        self.mask = mask
    
    def forward(self, inputs):
        pos = self.pos_encoding.forward(inputs)
        att = self.multi_head_attention.forward(pos)
        norm_att = self.norm.forward(pos, att)
        output = self.linear(norm_att)
        return output
    
    def batch_process(self, X, y):
        MSELoss = 0
        y_pred = self.forward(X).squeeze()
        loss = self.criterion(y_pred, y)
        MSELoss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return MSELoss

    def fit(self, train_loader, num_epochs = 200, learning_rate = 0.01, print_period = 20):
        # Define loss and optimizer
        self.criterion = nn.MSELoss()
        self.parameters = [
            *self.multi_head_attention.parameters(),
            *self.norm.parameters(),
            *self.linear.parameters()
        ]
        self.optimizer = optim.Adam(self.parameters, lr=learning_rate)
        for i in range(num_epochs):
            MSELoss = 0
            for batch in train_loader:
                X = batch['X']
                y = batch['y']
                MSELoss += self.batch_process(X, y)
            RMSELoss = np.sqrt(MSELoss/X.shape[0])
            if i%print_period == 0:
                print(f'Step: {i}')
                print(f"RMSE loss: {RMSELoss}")
    
    def predict(self, X, y):
        pass
