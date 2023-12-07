import pandas as pd
import numpy as np
from utils import get_season

# For data analysis, please take a look on my notebook

class DataSet:
    def __init__(self):
        pass

    def store_data(self, df, store):
        df_store = df[df['Store'] == store]
        return df_store
    
    def add_remove_features(self, df):
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], dayfirst=True)
        df_copy['Month'] = df_copy['Date'].dt.month
        df_copy['Year'] = df_copy['Date'].dt.year
        df_copy['Season'] = df_copy['Month'].apply(get_season)
        df_copy['Week'] = df_copy['Date'].dt.isocalendar().week.astype('int')
        df_copy.drop('Date', axis = 1, inplace = True)
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
            target = df.iloc[i + input_chunk, 1]  # Target for this week
    
            sequences.append(sequence)
            targets.append(target) 
        return sequences, targets   