import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention
from normalized_layer import NormalizedLayer

class SelfAttentionModule(nn.Module):
    def __init__(self, heads, input_size, input_length = 4, mask = True):
        super().__init__()
        # input_size is the dimension of inputs
        self.input_size = input_size
        self.input_length = input_length
        self.heads = heads
        self.pos_encoding = PositionalEncoding(input_size, input_length)
        self.multi_head_attention = MultiHeadAttention(input_size, input_length, heads, mask)
        self.norm = NormalizedLayer(input_size, input_length)
        self.linear = nn.Linear(input_size, 1)
        self.mask = mask
    
    def forward(self, inputs):
        pos = self.pos_encoding.forward(inputs)
        att, self.attention_score = self.multi_head_attention.forward(pos)
        norm_att = self.norm.forward(inputs, att)
        output = self.linear(norm_att)
        return output
    
    def process(self, x, yknown, y):
        MSELoss = 0
        # Remember the output of the self-attention layer is of size
        # (n_heads, input_chunk, n_features)
        # and via the linear layer, we actually project it to the shape
        # (input_chunk, 1): which tells us how the current time input_chunk
        # depends on previous inputs
        y_hat = self.forward(x)
        y_pred = (y_hat[:-1].reshape(1, -1) @ yknown).squeeze()/self.input_length
        loss = self.criterion(y_pred, y)
        MSELoss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return MSELoss

    def fit(self, X_train, yknown_train, y_train, num_epochs = 200, learning_rate = 0.01, print_period = 20):
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
                # X_train is now 3 dimensional
                # y_train is now 2 dimensional
            for x, yknown, y in zip(X_train, yknown_train, y_train):
                MSELoss += self.process(x, yknown, y)
            RMSELoss = np.sqrt(MSELoss/X_train.shape[0])
            if i%print_period == 0:
                print(f'Step: {i}')
                print(f"RMSE loss for training set: {RMSELoss}")
    
    def predict(self, X_test, yknown_test, y_test):
        MSELoss = 0
        test_results = []
        test_criterion = nn.MSELoss()

        for x, yknown, y in zip(X_test, yknown_test, y_test):
            # Not mean, should be the attention weights
            y_hat = self.forward(x)
            y_pred = (y_hat[:-1].reshape(1, -1) @ yknown).squeeze()/self.input_length
            loss = test_criterion(y_pred, y)
            MSELoss += loss.item()
            # This is to print or to plot
            y_pred_lst = y_pred.detach().numpy().tolist()
            test_results.append(y_pred_lst)

        RMSELoss = np.sqrt(MSELoss/X_test.shape[0])
        print(f"RMSE loss for test set: {RMSELoss}")
        return test_results