import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor

# Load the Boston House Price Data
house_price_dataset = pd.read_csv("BostonHouse.csv")

# Data Analysis
house_price_dataset.head()
n0 = house_price_dataset.isnull().sum()
n1 = house_price_dataset.describe()

# Understating the correlation
corr = house_price_dataset.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corr, cbar=True, cmap='Blues', square=True, fmt='.1f', annot=True, annot_kws={'size': 8} )
plt.title("Correlation Heatmap for Boston Housing Data")
plt.show()

# Splitting feature and target
X = house_price_dataset.drop(['price'], axis=1)
Y = house_price_dataset["price"]

# Train Test Spilt
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2, test_size=0.2)

# Model Training
model = XGBRegressor()
model.fit(X_train, Y_train)

# Training Evaluation
training_data_prediction = model.predict(X_train)
score_1 = metrics.r2_score(Y_train, training_data_prediction)
score_2 = metrics.mean_squared_error(Y_train, training_data_prediction)

print(f"Training R Square Error: {score_1} ")
print(f"Training Mean Square Error: {score_2}")

