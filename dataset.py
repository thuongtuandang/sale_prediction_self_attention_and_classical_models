import pandas as pd
import numpy as np
from utils import get_season

# For data analysis, please take a look on my notebook

class DataSet:
    def __init__(self):
        pass

    def read_data(self, data_path, store):
        df = pd.read_csv(data_path)
        df = df[df['Store'] == store]
        return df
    
    def add_remove_features(self, df):
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], dayfirst=True)
        df_copy['Month'] = df_copy['Date'].dt.month
        df_copy['Year'] = df_copy['Date'].dt.year
        df_copy['Season'] = df_copy['Month'].apply(get_season)
        df_copy['Week'] = df_copy['Date'].dt.isocalendar().week.astype('int')
        df_copy.drop('Date', axis = 1, inplace = True)
        df_copy.drop('Store', axis = 1, inplace = True)
        return df_copy

    def remove_outliers(self, df):
        df_copy = df.copy()
        lower_indices = df_copy[df_copy['Unemployment'] < 5].index
        upper_indices = df_copy[df_copy['Unemployment'] > 11].index
        df_copy.drop(lower_indices, axis = 0, inplace = True)
        df_copy.drop(upper_indices, axis = 0, inplace = True)
        return df_copy
    
    def prepare_data(self, df, input_chunk):
        sequences = []
        targets = []

        # Extract sequences and targets using a sliding window approach
        for i in range(len(df) - input_chunk):
            sequence = df.iloc[i : i + input_chunk].values  # Input sequence (4 previous weeks)
            target = df.iloc[i + input_chunk]  # Target for this week
    
            sequences.append(sequence)
            targets.append(target) 
        return sequences, targets   
    
    def train_test_split(self, X, y, train_size = 0.8):
        length = len(X)
        train_length = int(train_size * length)
        X_train = X[:train_length]
        X_test = X[train_length:]
        y_train = np.array(y[:train_length])
        y_train = y_train[:, 0]
        y_test = np.array(y[train_length:])
        y_test = y_test[:, 0]
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)