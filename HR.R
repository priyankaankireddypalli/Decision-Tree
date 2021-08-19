# 4
getOption("repos")
library(readr)
# Importing the Data
hr <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\HR_DT.csv')
# Performing EDA
str(hr)
table(hr$monthly.income.of.employee)
hr$monthly.income.of.employee <- as.factor(hr$monthly.income.of.employee)
hr$no.of.Years.of.Experience.of.employee <- as.factor(hr$no.of.Years.of.Experience.of.employee)
# Shuffle the data
hrrand <- hr[order(runif(196)), ]
str(hrrand)
# Splitting the data into train and test
train <- hrrand[1:98, ]
test  <- hrrand[99:196, ]
# Check the proportion of class variable
prop.table(table(hrrand$monthly.income.of.employee))
prop.table(table(train$monthly.income.of.employee))
prop.table(table(test$monthly.income.of.employee))
# Training a model on the data
library(C50)
model <- C5.0(train[, -3], train$monthly.income.of.employee)
windows()
plot(model) 
# Display detailed information about the tree
summary(model)
# Evaluating model performance
# Test data accuracy
testres <- predict(model, test)
testacc <- mean(test$monthly.income.of.employee == testres)
testacc
# Cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(test$monthly.income.of.employee, testres, dnn = c('actual default', 'predicted default'))
# On Training Dataset
trainres <- predict(model, train)
trainacc <- mean(train$monthly.income.of.employee == trainres)
trainacc
table(train$monthly.income.of.employee, trainres)
