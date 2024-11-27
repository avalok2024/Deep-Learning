import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor

# Load the House Price Data
house_price_dataset = pd.read_csv("BostonHouse.csv")

# Data Analysis
print(house_price_dataset.head())