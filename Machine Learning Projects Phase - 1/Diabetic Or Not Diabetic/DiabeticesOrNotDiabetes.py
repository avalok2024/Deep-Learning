from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from spacy.cli.train import train

# Data Collection and Analysis
diabetes_data = pd.read_csv("/Machine Learning Projects Phase - 1/Diabetic Or Not Diabetic/diabetes.csv")
# print(diabetes_data.head())

"How much total data are here."
# print(diabetes_data.shape)

# print(diabetes_data.describe())
"""0 - Non Diabetic 
1 - Diabetic"""
# print(diabetes_data['Outcome'].value_counts())

"Dividing/Grouping the Outcome in two as given Diabetic or Not Diabetic in mean form"
# print(diabetes_data.groupby('Outcome').mean())


"""Separating the labeled and unordered data"""
X = diabetes_data.drop(columns="Outcome", axis = 1)
Y = diabetes_data['Outcome']
# print(X)
# print(Y)

"Data Standardization"
scaler = StandardScaler()
scaler.fit(X)
standardiztion_data = scaler.transform(X)
# Or scaler.fit_transform(X)

X = standardiztion_data
Y = diabetes_data['Outcome']

# Test and Train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

"Check the X = X_test + X_train"
print(X, X_test.shape, X_train.shape)

# Classifier
classifier = svm.SVC(kernel='linear')

'training the support vector machine learning'
classifier.fit(X_train, Y_train)


# Model Evaluation
'Accuracy Score on the training data'

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy Score of Training data :", training_data_accuracy * 100 , " %")

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy Score of Test data: ', test_data_accuracy * 100, " %")

# Making Prediction System
input_data = (1,89,66,23,94,28.1,0.167,21)
input_data_as_array = np.asarray(input_data)
reshaped_array = input_data_as_array.reshape(1, -1)
scl_data = scaler.transform(reshaped_array)
# print(scl_data)

prediction = classifier.predict(scl_data)

print("_"*70)

if prediction[0] == "1":
    print("Diabetic!")
else:
    print("Not Diabetic")






