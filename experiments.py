import numpy as np
import pandas as pd
from dataset import DataSet
import torch
from self_attention_module import SelfAttentionModule
from custom_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils import pad_h_stack, pad_v_stack, reshape
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/Walmart.csv'
input_chunk = 16

# Load dataset
df_total = pd.read_csv(DATA_PATH)
ds = DataSet()
X_train_total = []
y_train_total = []
X_test_total = []
y_test_total = []

for store_number in range (1, 46, 1):
    df = ds.store_data(df_total, store = store_number)
    df = ds.add_remove_features(df)
    # df = ds.remove_outliers(df)

    # Fit transform numerical features
    num_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    sc = StandardScaler()
    df[num_features] = sc.fit_transform(df[num_features])

    # Create X, y
    # We will use data of 4 previous week to predict the sales of next week
    list_X, list_y = ds.prepare_data(df, input_chunk)

    # Train, test split
    X_train, X_test, y_train, y_test = train_test_split(list_X, list_y, test_size=0.2)
    
    # Append them to the total sets
    X_train_total += X_train
    y_train_total += y_train
    X_test_total += X_test
    y_test_total += y_test

X_train = np.array(X_train_total)
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = np.array(y_train_total)
y_train = torch.tensor(y_train, dtype=torch.float)
X_test = np.array(X_test_total)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = np.array(y_test_total)
y_test = torch.tensor(y_test, dtype=torch.float)


input_size = X_train.shape[2]
SelfAttModel = SelfAttentionModule(heads = 1, input_size = input_size, input_length = input_chunk, mask = True)
# input of fit is 4 dimensional arrays
SelfAttModel.fit(X_train, y_train, num_epochs = 21, print_period = 10, learning_rate = 0.015)

# # Test step
# y_pred = SelfAttModel.predict(X_test, y_test)

# # Plot the results in log space
# y_pred_np = y_pred[0].detach().numpy()
# for i in range(len(y_pred)-1):
#     y = y_pred[i+1].detach().numpy()
#     y_pred_np = np.hstack([y_pred_np, y])

# y_test = y_test.detach().numpy()

# plt.figure(figsize=(14, 6))
# length = len(y_pred)
# sns.lineplot(x = range(length), y = y_test, label = 'test values', color = "b")
# sns.lineplot(x = range(length), y = y_pred_np, label = 'predicted values', color = "r")
# plt.title('Plotting for store 1')
# plt.legend()
# plt.show()