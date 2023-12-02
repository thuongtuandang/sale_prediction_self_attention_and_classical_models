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
        df_copy['Day'] = df_copy['Date'].dt.day
        df_copy['Month'] = df_copy['Date'].dt.month
        df_copy['Year'] = df_copy['Date'].dt.year
        df_copy['Season'] = df_copy['Month'].apply(get_season)
        df_copy.drop('Date', axis = 1, inplace = True)
        return df_copy

    def remove_outliers(self, df):
        df_copy = df.copy()
        lower_indices = df_copy[df_copy['Unemployment'] < 5].index
        upper_indices = df_copy[df_copy['Unemployment'] > 11].index
        df_copy.drop(lower_indices, axis = 0, inplace = True)
        df_copy.drop(upper_indices, axis = 0, inplace = True)
        return df_copy
    
    def create_X_y(self, df):
        X = df.drop('Weekly_Sales', axis = 1).values
        y = df['Weekly_Sales'].values
        return X, y
    
    def train_test_split(self, X, y, test_size = 0.2, random_state = 42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

