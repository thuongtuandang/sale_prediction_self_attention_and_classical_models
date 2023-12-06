import numpy as np
import pandas as pd
from dataset import DataSet
import torch
from self_attention_module import SelfAttentionModule
from custom_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils import pad_h_stack, pad_v_stack
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/Walmart.csv'
input_chunk = 4
store_number = 5

# Load dataset
ds = DataSet()
df = ds.read_data(DATA_PATH, store = store_number)
df = ds.add_remove_features(df)
df = ds.remove_outliers(df)

# Fit transform numerical features
num_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
sc = StandardScaler()
df[num_features] = sc.fit_transform(df[num_features])

# Create X, y
# We will use data of 4 previous week to predict the sales of next week
list_X, list_y = ds.prepare_data(df, input_chunk)

# Train, test split
X_train, X_test, y_train, y_test = ds.train_test_split(list_X, list_y, train_size=0.8)

# Convert numpy arrays to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float, requires_grad=False)
y_train = torch.tensor(y_train, dtype=torch.float, requires_grad=False)
X_test = torch.tensor(X_test, dtype=torch.float, requires_grad=False)
y_test = torch.tensor(y_test, dtype=torch.float, requires_grad=False)

# Train step
input_size = X_train.shape[2]
SelfAttModel = SelfAttentionModule(heads = 1, input_size = input_size, input_length = input_chunk, mask = True)
SelfAttModel.fit(X_train, y_train, num_epochs = 201, print_period = 20, learning_rate = 0.015)

# Test step
y_pred = SelfAttModel.predict(X_test, y_test)

# Plot the results in log space
y_pred_np = y_pred[0].detach().numpy()
for i in range(len(y_pred)-1):
    y = y_pred[i+1].detach().numpy()
    y_pred_np = np.hstack([y_pred_np, y])

y_test = y_test.detach().numpy()

plt.figure(figsize=(14, 6))
length = len(y_pred)
sns.lineplot(x = range(length), y = y_test, label = 'test values', color = "b")
sns.lineplot(x = range(length), y = y_pred_np, label = 'predicted values', color = "r")
plt.title('Plotting for store 1')
plt.legend()
plt.show()