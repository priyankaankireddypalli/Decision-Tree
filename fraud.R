# 3
library(readr)
# Importing the dataset
Fraud_Data <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Fraud_check.csv')
View(Fraud_Data)
str(Fraud_Data)
# Performing EDA
Fraud_Data$Undergrad <- as.factor(Fraud_Data$Undergrad)
Fraud_Data$Marital.Status <- as.factor(Fraud_Data$Marital.Status)
Fraud_Data$Urban <- as.factor(Fraud_Data$Urban)
# We will use the If else statement to group the person getting salary as less than 30000 as risky and good.
risk <- ifelse(Fraud_Data$Taxable.Income <= 30000, "Good","Risky")
risk<-as.factor(risk)
fraud_new <- cbind(Fraud_Data,risk)
# Shuffle the data
fraud_data_rand <- fraud_new[order(runif(600)), ]
str(fraud_data_rand)
# Split the data into train and test
fraud_data_train <- fraud_data_rand[1:499, ]
fraud_data_test  <- fraud_data_rand[500:600, ]
# check the proportion of class variable
prop.table(table(fraud_new$risk))
prop.table(table(fraud_data_train$risk))
prop.table(table(fraud_data_test$risk))
# Training the model
install.packages("C50")
library(C50)
fraud_data_model <- C5.0(fraud_data_train[, -7], fraud_data_train$risk)
windows()
plot(fraud_data_model) 
# Display detailed information about the tree
summary(fraud_data_model)
# Evaluating the model performance
# Test data accuracy
test_res <- predict(fraud_data_model, fraud_data_test)
test_acc <- mean(fraud_data_test$risk == test_res)
test_acc
# Cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(fraud_data_test$Urban, test_res, dnn = c('actual default', 'predicted default'))
# On Training Dataset
train_res <- predict(fraud_data_model,fraud_data_train)
train_acc <- mean(fraud_data_train$risk == train_res)
train_acc
#we are getting train accuracy as 100% and test accuracy as 100%.

