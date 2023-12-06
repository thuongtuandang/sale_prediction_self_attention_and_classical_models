import pandas as pd
import numpy as np
from utils import get_season
from sklearn.model_selection import train_test_split

# For data analysis, please take a look on my notebook

class DataSet:
    def __init__(self):
        pass

    def read_data(self, data_path):
        df = pd.read_csv(data_path)
        return df
    
    def add_remove_features(self, df):
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], dayfirst=True)
        df_copy['Month'] = df_copy['Date'].dt.month
        df_copy['Year'] = df_copy['Date'].dt.year
        df_copy['Season'] = df_copy['Month'].apply(get_season)
        df_copy['Week'] = df_copy['Date'].dt.isocalendar().week
        df_copy.drop('Date', axis = 1, inplace = True)
        return df_copy

    def remove_outliers(self, df):
        df_copy = df.copy()
        lower_indices = df_copy[df_copy['Unemployment'] < 5].index
        upper_indices = df_copy[df_copy['Unemployment'] > 11].index
        df_copy.drop(lower_indices, axis = 0, inplace = True)
        df_copy.drop(upper_indices, axis = 0, inplace = True)
        return df_copy
    
    # We want our train 
    def train_test_split(self, df, test_size = 0.2, random_state = 42):
        train_percentage = 1 - test_size
        grouped = df.groupby('Store')
        train_data = []
        test_data = []
        for store, data in grouped:
            total_rows = len(data)
            train_size = int(total_rows * train_percentage)
    
            train_set = data.iloc[: train_size]
            test_set = data.iloc[train_size:]
    
            train_data.append(train_set)
            test_data.append(test_set)
        train_df = pd.concat(train_data)
        train_df = train_df.sample(frac = 1, random_state = random_state)
        test_df = pd.concat(test_data)
        test_df = test_df.sample(frac=1, random_state=random_state)
        return train_df, test_df

    
    def create_X_y(self, df):
        X = df.drop('Weekly_Sales', axis = 1).values
        y = df['Weekly_Sales'].values
        return X, y

