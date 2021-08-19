# 3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Importing the dataset
fraud = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Fraud_check.csv")
# Checking for null values
fraud.isnull().sum()
fraud.columns
# Converting into binary
lb = LabelEncoder()
fraud["Undergrad"] = lb.fit_transform(fraud["Undergrad"])
fraud["Marital.Status"] = lb.fit_transform(fraud["Marital.Status"])
fraud["Taxable.Income"] = lb.fit_transform(fraud["Taxable.Income"])
fraud["City.Population"] = lb.fit_transform(fraud["City.Population"])
fraud["Work.Experience"] = lb.fit_transform(fraud["Work.Experience"])
fraud["Urban"] = lb.fit_transform(fraud["Urban"])
fraud['Taxable.Income'].unique()
fraud['Taxable.Income'].value_counts()
colnames = list(fraud.columns)
predictors = colnames[:6]
target = colnames[5]
# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(fraud, test_size = 0.3)
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
