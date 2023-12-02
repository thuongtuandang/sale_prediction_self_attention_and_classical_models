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