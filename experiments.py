import numpy as np
import pandas as pd
from dataset import DataSet
import torch
from self_attention_module import SelfAttentionModule
from custom_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils import pad_h_stack, pad_v_stack, reshape, torch_convert
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle

DATA_PATH = 'data/Walmart.csv'
input_chunk = 4

# Load dataset
df_total = pd.read_csv(DATA_PATH)
ds = DataSet()
X_train_total = []
y_train_total = []
yknown_train_total = []
yknown_test_total = []
X_test_total = []
y_test_total = []

for store_number in range (1, 10, 1):
    df = ds.store_data(df_total, store = store_number)
    df = ds.add_remove_features(df)
    # df = ds.remove_outliers(df)

    # Fit transform numerical features
    num_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    sc = StandardScaler()
    df[num_features] = sc.fit_transform(df[num_features])

    # Create X, y
    # We will use data of 4 previous week to predict the sales of next week
    list_X, list_yknown, list_y = ds.prepare_data(df, input_chunk)
    # Zip X and yknown
    zipped_data = list(zip(list_X, list_yknown))

    # Train, test split
    X_train_yknown, X_test_yknown, y_train, y_test = train_test_split(zipped_data, list_y, shuffle = False, test_size=0.2)
    X_train, yknown_train = zip(*X_train_yknown)
    X_test, yknown_test = zip(*X_test_yknown)

    # Append them to the total sets
    X_train_total += X_train
    yknown_train_total += yknown_train
    y_train_total += y_train
    X_test_total += X_test
    yknown_test_total += yknown_test
    y_test_total += y_test

# Combine X_train_total and yknown_train_total into a single array for shuffling
combined_data = list(zip(X_train_total, yknown_train_total))
# Shuffle the combined data using the same random_state to maintain correspondence
shuffled_combined_data = shuffle(combined_data, random_state=42)
# Unzip the shuffled data back into separate arrays
X_train_total, yknown_train_total = zip(*shuffled_combined_data)

# Convert all array to torch
X_train = torch_convert(X_train_total)
yknown_train = torch_convert(yknown_train_total)
y_train = torch_convert(y_train_total)

X_test = torch_convert(X_test_total)
yknown_test = torch_convert(yknown_test_total)
y_test = torch_convert(y_test_total)


input_size = X_train.shape[2]
SelfAttModel = SelfAttentionModule(heads = 1, input_size = input_size, input_length = input_chunk + 1, mask = True)
# # input of fit is 3 dimensional array for X
SelfAttModel.fit(X_train, yknown_train, y_train, num_epochs = 101, print_period = 1, learning_rate = 1e-6)

# Test step and plot results
y_pred = SelfAttModel.predict(X_test, yknown_test, y_test)
y_test_lst = y_test.detach().numpy().tolist()

plt.figure(figsize=(14, 6))
# length = len(y_pred)
length = 100
sns.lineplot(x = range(length), y = y_test_lst[60:160], label = 'test values', color = "b")
sns.lineplot(x = range(length), y = y_pred[60:160], label = 'predicted values', color = "r")
plt.title('Plotting the results')
plt.legend()
plt.show()