# 2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Importing the dataset
diab = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Diabetes.csv")
# Checking for null values
diab.isnull().sum()
diab.columns
# Converting into binary
lb = LabelEncoder()
diab[" Number of times pregnant"] = lb.fit_transform(diab[" Number of times pregnant"])
diab[" Plasma glucose concentration"] = lb.fit_transform(diab[" Plasma glucose concentration"])
diab[" Diastolic blood pressure"] = lb.fit_transform(diab[" Diastolic blood pressure"])
diab[" Triceps skin fold thickness"] = lb.fit_transform(diab[" Triceps skin fold thickness"])
diab[" 2-Hour serum insulin"] = lb.fit_transform(diab[" 2-Hour serum insulin"])
diab[" Body mass index"] = lb.fit_transform(diab[" Body mass index"])
diab[" Diabetes pedigree function"] = lb.fit_transform(diab[" Diabetes pedigree function"])
diab[" Age (years)"] = lb.fit_transform(diab[" Age (years)"])
diab[" Class variable"] = lb.fit_transform(diab[" Class variable"])
colnames = list(diab.columns)
predictors = colnames[:9]
target = colnames[8]
# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(diab, test_size = 0.3)
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])
# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])
np.mean(preds == test[target]) # Test Data Accuracy 
# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])
np.mean(preds == train[target]) # Train Data Accuracy
