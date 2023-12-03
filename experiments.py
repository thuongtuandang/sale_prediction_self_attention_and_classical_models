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

DATA_PATH = 'data/Walmart.csv'
batch_length = 128

# Load dataset
ds = DataSet()
df = ds.read_data(DATA_PATH)
df = ds.add_remove_features(df)
df = ds.remove_outliers(df)

# Fit transform numerical features
num_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
sc = StandardScaler()
df[num_features] = sc.fit_transform(df[num_features])

# Create X, y
X, y = ds.create_X_y(df)
X_train, X_test, y_train, y_test = ds.train_test_split(X, y)

# Log transform y_train
y_train = np.log(y_train)

# Padding zero rows such that n_rows is divisible by batch_size
X_train = pad_v_stack(X_train, X_train[X_train.shape[0] - 1], batch_length)
y_train = pad_h_stack(y_train, y_train[y_train.shape[0] - 1], batch_length)


# Train step
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_length)

input_size = X.shape[1]
SelfAttModel = SelfAttentionModule(heads = 2, input_size=input_size, batch_length = batch_length)
SelfAttModel.fit(train_loader=train_loader, num_epochs=501, print_period=20, learning_rate=0.03)

# Test step
X_test = pad_v_stack(X_test, X_test[X_test.shape[0]-1], batch_length)
y_test = pad_h_stack(y_test, y_test[y_test.shape[0]-1], batch_length)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)
y_pred = SelfAttModel.predict(X_test, y_test)

y_pred_np = y_pred[0].detach().numpy()
for i in range(len(y_pred)-1):
    y = y_pred[i+1].detach().numpy()
    y_pred_np = np.hstack([y_pred_np, y])

y_test = y_test.detach().numpy()

plt.figure(figsize=(14, 6))
ax1 = sns.distplot(y_test, hist = False, color = "b")
ax2 = sns.distplot(y_pred_np, hist = False, color = "r", ax = ax1)
plt.legend()
plt.show()