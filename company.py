# 1
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Importing the dataset
comp = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Company_Data.csv")
# Checking for null values
comp.isnull().sum()
comp.columns
# Converting into binary
lb = LabelEncoder()
comp["Sales"] = lb.fit_transform(comp["Sales"])
comp["CompPrice"] = lb.fit_transform(comp["CompPrice"])
comp["Income"] = lb.fit_transform(comp["Income"])
comp["Advertising"] = lb.fit_transform(comp["Advertising"])
comp["Population"] = lb.fit_transform(comp["Population"])
comp["Price"] = lb.fit_transform(comp["Price"])
comp["ShelveLoc"] = lb.fit_transform(comp["ShelveLoc"])
comp["Age"] = lb.fit_transform(comp["Age"])
comp["Education"] = lb.fit_transform(comp["Education"])
comp["Urban"] = lb.fit_transform(comp["Urban"])
comp["US"] = lb.fit_transform(comp["US"])
comp['Sales'].unique()
comp['Sales'].value_counts()
colnames = list(comp.columns)
predictors = colnames[:11]
target = colnames[0]
# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(comp, test_size = 0.3)
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])
# Prediction on Test Data
pred = model.predict(test[predictors])
pd.crosstab(test[target], pred, rownames=['Actual'], colnames=['Predictions'])
np.mean(pred == test[target]) # Test Data Accuracy 
# Prediction on Train Data
pred = model.predict(train[predictors])
pd.crosstab(train[target], pred, rownames = ['Actual'], colnames = ['Predictions'])
np.mean(pred == train[target]) # Train Data Accuracy
