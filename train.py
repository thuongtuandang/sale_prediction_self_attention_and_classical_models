import torch
import torch.nn as nn
import torch.optim as optim
from self_attention_module import SelfAttentionModule

class Train():
    def __init__(self, input_size, heads = 1, mask = True):
        super().__init__()
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD()
        self.model = SelfAttentionModule(input_size, heads, mask)
    
    def process(self, X_train, y_train):
        for 