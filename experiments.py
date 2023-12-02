import numpy as np
import pandas as pd
from dataset import DataSet
import torch
from self_attention_module import SelfAttentionModule
from custom_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader

DATA_PATH = 'data/Walmart.csv'

# Load dataset
ds = DataSet()
df = ds.read_data(DATA_PATH)
df = ds.add_remove_features(df)
df = ds.remove_outliers(df)
X, y = ds.create_X_y(df)
X_train, X_test, y_train, y_test = ds.train_test_split(X, y)

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

input_size = X.shape[1]
batch_length = 1
SelfAttModel = SelfAttentionModule(heads = 1, input_size=input_size, batch_length=batch_length)
SelfAttModel.fit(train_loader=train_loader, num_epochs=201, print_period=20, learning_rate=10)