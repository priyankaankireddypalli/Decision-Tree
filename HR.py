# 4
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Importing the dataset
hr = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\HR_DT.csv")
# Checking for null values
hr.isnull().sum()
hr.columns
# Dropping passed columns 
hr.drop(["Position of the employee"], axis = 1, inplace = True) 
# display 
hr 
# Converting into binary
lb = LabelEncoder()
hr["no of Years of Experience of employee"] = lb.fit_transform(hr["no of Years of Experience of employee"])
hr[" monthly income of employee"] = lb.fit_transform(hr[" monthly income of employee"])
hr[' monthly income of employee'].unique()
hr[' monthly income of employee'].value_counts()
colnames = list(hr.columns)
predictors = colnames[:2]
target = colnames[1]
# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(hr, test_size = 0.3)
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
