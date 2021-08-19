# 1
library(readr)
# Importing the Data
comp <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Company_Data.csv')
str(comp)
# Splitting the data into train and test
library(caTools)
set.seed(0)
split <- sample.split(comp$Sales, SplitRatio = 0.8)
train <- subset(comp, split == TRUE)
test <- subset(comp, split == FALSE)
library(rpart)
model <- rpart(train$Sales ~ ., data = train,control = rpart.control(cp = 0, maxdepth = 3))
# Plot Decision Tree
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(model, box.palette = "auto", digits = -3)
# Measure the RMSE on Test data
testpred <- predict(model, newdata = test, type = "vector")
# RMSE
accuracy1 <- sqrt(mean(test$Sales - testpred)^2)
accuracy1
# Measure the RMSE on Train data
trainpred <- predict(model, newdata = train, type = "vector")
# RMSE
accuracy2 <- sqrt(mean(train$Sales - trainpred)^2)
accuracy2
# Prune the Decision Tree
# Grow the full tree
fullmodel <- rpart(train$Sales ~ ., data = train,control = rpart.control(cp = 0))
rpart.plot(fullmodel, box.palette = "auto", digits = -3)
# Examine the complexity plot
# Tuning parameter check the value of cp which is giving us minimum cross validation error (xerror)
printcp(fullmodel)   
plotcp(model)
mincp <- model$cptable[which.min(model$cptable[, "xerror"]), "CP"]
# Prune the model based on the optimal cp value
model_pruned_1 <- prune(fullmodel, cp = mincp)
rpart.plot(model_pruned_1, box.palette = "auto", digits = -3)
model_pruned_2 <- prune(fullmodel, cp = 0.02)
rpart.plot(model_pruned_2, box.palette = "auto", digits = -3)
# Measure the RMSE using Full tree
test_pred_fultree <- predict(fullmodel, newdata = test, type = "vector")
# RMSE
accuracy_f <- sqrt(mean(test$Sales - test_pred_fultree)^2)
accuracy_f
# Measure the RMSE using Prune tree - model1
test_pred_prune1 <- predict(model_pruned_1, newdata = test, type = "vector")
# RMSE
accuracy_prune1 <- sqrt(mean(test$Sales - test_pred_prune1)^2)
accuracy_prune1
# Measure the RMSE using Prune tree - model2
test_pred_prune2 <- predict(model_pruned_2, newdata = test, type = "vector")
# RMSE
accuracy_prune2 <- sqrt(mean(test$Sales - test_pred_prune2)^2)
accuracy_prune2
# Prediction for trained data result
train_pred_fultree <- predict(fullmodel,train, type = 'vector')
# RMSE on Train Data
train_accuracy_fultree <- sqrt(mean(train$Sales - train_pred_fultree)^2)
train_accuracy_fultree
# Prediction for trained data result
train_pred_prune1 <- predict(model_pruned_1, train, type = 'vector')
# RMSE on Train Data
train_accuracy_fultree2 <- sqrt(mean(train$Sales - train_pred_prune1)^2)
train_accuracy_fultree2
# Prediction for trained data result
train_pred_prune2 <- predict(model_pruned_2,train, type = 'vector')
# RMSE on Train Data
train_accuracy_fultree2 <- sqrt(mean(train$Sales - train_pred_prune2)^2)
train_accuracy_fultree2

