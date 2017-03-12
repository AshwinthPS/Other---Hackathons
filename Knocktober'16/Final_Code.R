########################################################   Knocktober'16 #######################################################
## A very Quick Code and Submission.
## author - AshwinthPS and Shashank Hegde
## Eval Metric  - ROC -AUC
## Score - 0.6899
## Rank - 120

setwd("C:\\Users\\HP\\Knocktober")
getwd()

raw_data <- read.csv("Train_Merged 3.csv",na.strings = "",stringsAsFactors = T)
pred_data<-read.csv("Test_Merged.csv",na.strings = "",stringsAsFactors = T)
library(caret)
library(ROCR)


#Split the training data in to two parts (75% train / 25% test)------------
set.seed(123)
indexes = createDataPartition(raw_data$Fav.Outcome,p = 0.75,list = FALSE)
train = as.data.frame(raw_data[indexes,]) 
test =as.data.frame (raw_data[-indexes,])


raw_data1<-merge(raw_data,cnt_train,by="Patient_ID")  

str(raw_data)
str(raw_data1)
str(train)
str(test)

# EDA------------
library(dplyr)
cnt_train<-train %>% group_by(Patient_ID) %>% summarise(Reg_Count = n())
train<-merge(train,cnt_train,by="Patient_ID")  
table(train$Reg_Count,train$Fav.Outcome)

cnt_test<-test %>% group_by(Patient_ID) %>% summarise(Reg_Count = n())
test<-merge(test,cnt_test,by="Patient_ID")  


cnt_pred<-pred_data %>% group_by(Patient_ID) %>% summarise(Reg_Count = n())
pred_data<-merge(pred_data,cnt_pred,by="Patient_ID")  
colnames(train)
colnames(test)
colnames(train)


#Predicting on test data------
test_PCA <- predict(Var_PCA,newdata = test)
pcs_test<-as.data.frame(test_PCA)
test<-data.frame(test$Fav.Outcome,pcs_test)
pred_logistic <- predict(mod,pcs_test)
summary(pred_logistic)

library(psych)
pairs.panels(train)

#Run Logistic regression for intial submission----------------- 
mod<-glm(Fav.Outcome~Health_Camp_ID+Var1+Var5+online+Category2+City_Type+Camp.Duration,family = binomial(link='logit'),data = train)
summary(mod)

pred<-predict(mod,newdata=subset(test,select=c(5,8,12,13,14,16,17)),type='response')
summary(pred)

#confusion matrix----------------- 
a<-ifelse(pred_logistic>0.24790,'1','0')
table(test$Fav.Outcome,pred>0.2495)

#miss Classification Error rate Logistic-------------- 
pred <- ifelse(pred > 0.24790,1,0)
misClasificError <- mean(pred != test$Fav.Outcome)
print(paste('Accuracy',1-misClasificError))

#Plot ROC logistic------------------

library(ROCR)
p <- predict(mod, newdata=subset(test,select=c(5,8,12,13,14,16,17)), type="response")
pr <- prediction(p, test$Fav.Outcome)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc



#predict on test data for final submission --------------
submission <- predict(mod,newdata=subset(pred_data,select=c(5,8,9,10,11,12,13)),type='response')
write.csv(submission,file = "finalprediction_hello.csv")
getwd()

library(caret)
library(C50)
library(irr)

set.seed(123)

folds <- createFolds(raw_data$Fav.Outcome, k = 2)


cv_results <- lapply(folds, function(x) {
  
  train = as.data.frame(raw_data[-x,]) 
  test =as.data.frame (raw_data[x,])
  
  mod<-glm(Fav.Outcome~Health_Camp_ID+Var1+Var5+online+Category2+City_Type+Camp.Duration,
           family = binomial(link='logit'),data = train)
  
  #pred<-predict(mod,newdata=subset(test,select=c(5,8,12,13,14,16,17)),type='response')
  return(mod)
}

)
 
cv_results

mod<-cv_results$Fold2

submission <- predict(final_model111,newdata=subset(pred_data,select=c(5,8,9,10,11,12,13)),type='response')
write.csv(submission,file = "finalprediction12.csv")
getwd()
