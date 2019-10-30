library(SuperLearner)
library(R.matlab)
library(ggplot2)
set.seed(10)

#############################################################################
# data: outcome must be labeled "Type" here with binary outcome
# n: initial number of samples from each category
# m: further step size
# query: pick samples based on the committee or final ensemble probability
# tot: number of samples you are willing to label
# SL.lib: SuperLearner library
#############################################################################

SL_activeLearner<-function(data,n,m,query,tot,SL.lib){
  
  num_samples<-NULL
  all_accuracy<-NULL
  
  data<-data[complete.cases(data),]
  
  data_T<-data[data$Type==1,]
  data_N<-data[data$Type==0,]
  
  #Shuffle
  data_T <- data_T[sample(nrow(data_T)),]
  data_N <- data_N[sample(nrow(data_N)),]
  
  #Randomly pick n samples from each category to train on
  data_N_train<-data_N[1:n,]
  data_T_train<-data_T[1:n,]
  
  #Construct test set
  data_N_test <- data_N[!(row.names(data_N) %in% row.names(data_N_train)),]
  data_T_test <- data_T[!(row.names(data_T) %in% row.names(data_T_train)),]
  test<-rbind.data.frame(data_N_test,data_T_test)
  
  #Construct training set
  train<-rbind.data.frame(data_N_train,data_T_train)

  X=train[,-1]
  Y=train[,1]
  newdata<-test[,-1]
  
  SL_out<-SuperLearner(Y=Y, X=X, newX=newdata, SL.library = SL.lib, family="binomial", cvControl=list(V=3), method="method.NNLS")
  Risk<-cbind.data.frame(row.names(data.frame(SL_out$cvRisk)),SL_out$cvRisk)
  Risk[,1] <- as.character(Risk[,1])
  num<-data.frame(table(is.na(Risk[,2]))["FALSE"])
  
  if(num[1,1]<length(SL.lib))
  {
    usable=Risk[!is.na(Risk[,2]),1]
    usable <- as.character(usable)
    SL.lib_usable <- sapply(strsplit(usable, split='_', fixed=TRUE), function(x) (x[1]))
    
    SL_out<-SuperLearner(Y=Y, X=X, newX=newdata, SL.library = SL.lib_usable, family="binomial", cvControl=list(V=3), method="method.NNLS")
    
  }else{
    SL_out=SL_out
  }
  
  if(query=="committee"){
    
    ###############
    #By committee:
    ###############
    
    lib_pred<-round(SL_out$library.predict)
    ave<-rowMeans(lib_pred, na.rm = FALSE, dims = 1)
    lib_pred<-cbind.data.frame(lib_pred,ave)
    
    lib_pred$sort = abs(lib_pred$ave - 0.5)
    lib_pred<-lib_pred[order(lib_pred$sort, decreasing=FALSE),]
    
    #Pick top samples algorithms have issues with (by committee):
    new<-row.names(lib_pred)[1:m]
  
  }else if(query=="probability"){
    
    ########################
    #By ensemble prediction:
    ########################
    
    pred<-data.frame(SL_out$SL.predict)
    
    #For testing purposes while we have labeled data:
    pred2<-data.frame(round(pred))
    
    pred$sort = abs(pred[,1] - 0.5)
    pred<-pred[order(pred$sort, decreasing=FALSE),]
    names(pred)<-c("Predicted", "Uncertain")
    
    #Pick top samples algorithms have issues with (by ensemble prediction):
    new<-row.names(pred)[1:m]
  }
  
  ##########################################################  
  #Accuracy and Learning Curve (since we have labeled data):
  ##########################################################
  
  pred2$match<-test$Type[match(row.names(pred2),row.names(test))]
  names(pred2)<-c("Predicted", "Actual")
  pred2$accuracy<-abs(pred2$Predicted-pred2$Actual)
  names(pred2)[3]<-"match"
  accuracy<-(1-(sum(pred2$match)/nrow(pred2)))*100
  
  n=2*n
  num_samples<-cbind(num_samples,n)
  all_accuracy<-cbind(all_accuracy,accuracy)
  
  total<-n+m
  
  while(total<=tot){
    
    #Add the most uncertain samples to the mix:
    train<-rbind.data.frame(train,test[match(new,row.names(test)),])
    test<-test[!row.names(test) %in% new, ]
    
    X=train[,-1]
    Y=train[,1]
    newdata<-test[,-1]
    
    SL_out<-SuperLearner(Y=Y, X=X, newX=newdata, SL.library = SL.lib, family="binomial", cvControl=list(V=3), method="method.NNLS")
    Risk<-cbind.data.frame(row.names(data.frame(SL_out$cvRisk)),SL_out$cvRisk)
    Risk[,1] <- as.character(Risk[,1])
    num<-data.frame(table(is.na(Risk[,2]))["FALSE"])
    
    if(num[1,1]<length(SL.lib))
    {
      usable=Risk[!is.na(Risk[,2]),1]
      usable <- as.character(usable)
      SL.lib_usable <- sapply(strsplit(usable, split='_', fixed=TRUE), function(x) (x[1]))
      
      SL_out<-SuperLearner(Y=Y, X=X, newX=newdata, SL.library = SL.lib_usable, family="binomial", cvControl=list(V=3), method="method.NNLS")
      SL_out
      
    }else{
      SL_out=SL_out
    }
    
    if(query=="committee"){
      
      ###############
      #By committee:
      ###############
      
      lib_pred<-round(SL_out$library.predict)
      ave<-rowMeans(lib_pred, na.rm = FALSE, dims = 1)
      lib_pred<-cbind.data.frame(lib_pred,ave)
      
      lib_pred$sort = abs(lib_pred$ave - 0.5)
      lib_pred<-lib_pred[order(lib_pred$sort, decreasing=FALSE),]
      
      #Pick top samples algorithms have issues with (by committee):
      new<-row.names(lib_pred)[1:m]
      
    }else if(query=="probability"){
      
      ########################
      #By ensemble prediction:
      ########################
      
      pred<-data.frame(SL_out$SL.predict)
      
      #For testing purposes while we have labeled data:
      pred2<-data.frame(round(pred))
      
      pred$sort = abs(pred[,1] - 0.5)
      pred<-pred[order(pred$sort, decreasing=FALSE),]
      names(pred)<-c("Predicted", "Uncertain")
      
      #Pick top samples algorithms have issues with (by ensemble prediction):
      new<-row.names(pred)[1:m]
    }
    
    ##########################################################  
    #Accuracy and Learning Curve (since we have labeled data):
    ##########################################################
    
    pred2$match<-test$Type[match(row.names(pred2),row.names(test))]
    names(pred2)<-c("Predicted", "Actual")
    pred2$accuracy<-abs(pred2$Predicted-pred2$Actual)
    names(pred2)[3]<-"match"
    accuracy<-(1-(sum(pred2$match)/nrow(pred2)))*100
    
    num_samples<-cbind(num_samples,total)
    all_accuracy<-cbind(all_accuracy,accuracy)
    
    total<-total+m
  }
  
  return(list(pred,num_samples,all_accuracy))

}

data<-read.csv("~/Dropbox/Berkeley_Projects/ActiveLearning/7-6-16/data_7_6_16.csv", header=TRUE)
SL.lib = c("SL.bayesglm","SL.polymars","SL.earth","SL.glm.interaction","SL.nnet","SL.gam","SL.glmnet","SL.rpartPrune","SL.stepAIC","SL.knn","SL.glm","SL.step","SL.svm")

out<-SL_activeLearner(data,n=10,m=10,query="probability",tot=100,SL.lib)

#Plot the learning curve
LC<-cbind.data.frame(t(out[[2]]),t(out[[3]]))
names(LC)<-c("Samples", "Accuracy")

p<-ggplot(LC, aes(x=Samples, y=Accuracy)) +  geom_line() + ggtitle("Learning Curve") + labs(x="Number of Samples", y="Accuracy")
setwd("~/Google Drive/CS289 project/SuperLearner/7-7-16")
ggsave("SuperLearner_LearningCurve_100samples.png", p)

pred<-data.frame(out[[1]])
pred<-cbind.data.frame(row.names(pred),round(pred$Predicted), pred)
names(pred)<-c("Sample", "SL Prediction", "SL Probability", "Uncertainty")

write.table(pred,"SuperLearner_Prediction_100samples.csv", row.names = FALSE, sep=",")


