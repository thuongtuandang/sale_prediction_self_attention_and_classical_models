import pandas as pd
import numpy as np

def get_season(month):
        if month in [12, 1, 2]:
            # This is winter
            return 3
        elif month in [3, 4, 5]:
            # This is spring
            return 0
        elif month in [6, 7, 8]:
            # This is summer
            return 1
        else:
            # This is autumn
            return 2

def pad_v_stack(X, vector, batch_size):
     X_copy = X.copy()
     length, _ = X.shape
     if len(X) % batch_size == 0:
          return X_copy
     else:
          for i in range(batch_size):
               if (length + i)% batch_size == 0:
                    return X_copy
               X_copy = np.vstack([X_copy, vector])

def pad_h_stack(y, vector, batch_size):
     y_copy = y.copy()
     length = y.shape[0]
     if len(y) % batch_size == 0:
          return y_copy
     else:
          for i in range(batch_size):
               if (length + i) % batch_size == 0:
                    return y_copy
               y_copy = np.hstack([y_copy, vector])