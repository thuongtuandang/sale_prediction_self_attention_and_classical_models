import numpy as np
import pandas as pd
from dataset import DataSet
import torch
from self_attention_module import SelfAttentionModule

DATA_PATH = 'data/Walmart.csv'

ds = DataSet()
df = ds.read_data(DATA_PATH)
df = ds.add_remove_features(df)
df = ds.remove_outliers(df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = ds.create_X_y(df)
X = torch.tensor(X).float()
y = torch.tensor(y).float()
X, y = X.to(device), y.to(device)

input_size = X.shape[1]
SelfAttModel = SelfAttentionModule(input_size, heads = 1, mask = True)
SelfAttModel.to(device)
SelfAttModel.fit(X,y, num_epochs=50, print_period=1, learning_rate=1000)