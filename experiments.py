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

# Train, test split
train_df, test_df = ds.train_test_split(df, test_size=0.2, random_state=42)

# Create X, y
X_train, y_train = ds.create_X_y(train_df)
X_test, y_test = ds.create_X_y(test_df)

# Log transform y_train
y_train = np.log(y_train)

# Padding zero rows such that n_rows is divisible by batch_size
X_train = pad_v_stack(X_train, X_train[X_train.shape[0] - 1], batch_length)
y_train = pad_h_stack(y_train, y_train[y_train.shape[0] - 1], batch_length)

# Train step
X_train = X_train.astype(np.float16)
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_length)

input_size = X_train.shape[1]
SelfAttModel = SelfAttentionModule(heads = 5, input_size=input_size, batch_length = batch_length)
SelfAttModel.fit(train_loader=train_loader, num_epochs = 301, print_period = 20, learning_rate = 0.01)

# Test step
X_test = pad_v_stack(X_test, X_test[X_test.shape[0]-1], batch_length)
y_test = pad_h_stack(y_test, y_test[y_test.shape[0]-1], batch_length)
X_test = X_test.astype(np.float16)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)
y_pred = SelfAttModel.predict(X_test, y_test)

# Plot the results in log space
y_test = torch.log(y_test)
y_pred_np = y_pred[0].detach().numpy()
for i in range(len(y_pred)-1):
    y = y_pred[i+1].detach().numpy()
    y_pred_np = np.hstack([y_pred_np, y])

y_test = y_test.detach().numpy()

plt.figure(figsize=(14, 6))
sns.lineplot(x = range(200), y = y_test[300:500], label = 'test values', color = "b")
sns.lineplot(x = range(200), y = y_pred_np[300:500], label = 'predicted values', color = "r")
plt.title('Plotting in log space')
plt.legend()
plt.show()