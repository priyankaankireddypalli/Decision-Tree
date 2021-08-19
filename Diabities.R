# 2
library(readr)
# Importing the Data
diab <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Diabetes.csv')

# Performing EDA
str(diab)
# look at the class variable
table(diab$Class.variable)
diab$Class.variable <- as.factor(diab$Class.variable)
diab$Number.of.times.pregnant <- as.factor(diab$Number.of.times.pregnant)
# Shuffle the data
diabrand <- diab[order(runif(768)), ]
str(diabrand)
# Splitting the data into train and test
train <- diabrand[1:384, ]
test  <- diabrand[385:768, ]
# Check the proportion of class variable
prop.table(table(diabrand$Class.variable))
prop.table(table(train$Class.variable))
prop.table(table(test$Class.variable))
# Training the model on the data
library(C50)
model <- C5.0(train[, -9], train$Class.variable)
windows()
plot(model) 
# Display detailed information about the tree
summary(model)
# Evaluating model performance
# Test data accuracy
testres <- predict(model, test)
testacc <- mean(test$Class.variable == testres)
testacc
# Cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(test$Class.variable, testres, dnn = c('actual default', 'predicted default'))
# On Training Dataset
trainres <- predict(model, train)
trainacc <- mean(train$Class.variable == trainres)
trainacc
table(train$Class.variable, trainres)
