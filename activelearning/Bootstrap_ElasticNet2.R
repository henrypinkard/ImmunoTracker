library("lattice")
library(dendextend)
library(glmnet)
library(ROCR)
require(ggplot2)

set.seed(100)
data<-read.csv("~/Dropbox/Berkeley_Projects/ActiveLearning/TCell.csv", sep=",", header=FALSE)


all_predictors<-data.frame(rep(0,ncol(data)))
names(all_predictors)<-"new"

all_accuracy<- rep(NA, 100)
header<-data.frame(names(data))
names(header)<-"Predictors"

for (j in 1:100)
{
  #Shuffle
  data <- data[sample(nrow(data)),]
  
  data$Type<-as.factor(data$Type)
  levels(data$Type)<-c('N','Y')
  
  data_N<-data[grep("N", data$Type), ]
  data_Y<-data[grep("Y", data$Type), ]
  
  data_N<-data_N[1:250,]
  
  data<-rbind.data.frame(data_N,data_Y)
  
  condition_dn2_1<-cbind.data.frame(row.names(data),data$Type)
  names(condition_dn2_1)<-c("Sample","Type")
  
  x=model.matrix(Type~.,data)[,-1]
  y=data$Type
  
  #Implement the glmnet() function over a grid of values ranging from 10^10 to 10^-2
  grid=10^seq(10,-2,length=1000)
  
  all_lambda<- rep(NA, 100)
  all_cvm<- rep(NA, 100)
  all_mce<- rep(NA, 100)
  all_compare<-rep(NA,200)
  
  pre<-rep(NA,nrow(data))
  pred<-matrix(rep(pre,100), ncol = 100)
  
  all_out2<-cbind.data.frame(condition_dn2_1,pred)
  
  for (i in 1:100)
  { 
    #Pick random sample from each category
    rows <- 1:length(y)
    test<-tapply(rows, y, function(x) sample(x, 5)) 
    train<-(-unlist(test))
    y.test=y[unlist(test)]
    
    #Fit the training set
    lasso.mod=glmnet(x[train,],y[train],alpha=0.5,lambda=grid, family = "binomial")
    
    #5-fold cross-validation with misclassification error
    cv.out=cv.glmnet(x[train,],y[train],alpha=1,family = "binomial",type.measure = "class", nfolds=5)
    
    #Lambda value with the minimum mean cross-validation error
    bestlam=cv.out$lambda.min
    
    out2<-data.frame(predict(lasso.mod,type="response", newx=x[unlist(test),], s=bestlam))
    
    all_out2[,names(all_out2)[i+2]] <- out2$X1[match(all_out2$Sample, row.names(out2) )]
    
    lambda<-as.data.frame(cv.out$lambda)
    cvm<-as.data.frame(cv.out$cvm)
    
    index <- as.numeric(which(lambda==bestlam))
    
    #Corresponding cross-validation error for the lambda.min (bestlam)
    error<-cvm[index,]
    
    all_lambda[i]<-bestlam
    all_cvm[i]<-error
    
    #Compute the associated test error 
    lasso.pred=predict(lasso.mod,s=bestlam,newx=x[unlist(test),],family = "binomial", type = "class")
    lasso.pred<-as.data.frame(lasso.pred)
    colnames(lasso.pred)<-c("Predicted")
    
    truth<-condition_dn2_1[condition_dn2_1$Sample %in% row.names(lasso.pred),]
    score<- rep(NA, 2)
    prediction<-rep(NA, 2)
    compare<-as.data.frame(cbind.data.frame(truth$Sample,truth$Type,prediction,score))
    compare$prediction <- lasso.pred$Predicted[match(compare[,1], row.names(lasso.pred))]
    
    compare[,2] <- factor(compare[,2], levels=c("Y", "N"))
    compare[,3] <- factor(compare[,3], levels=c("Y", "N"))
    
    for (p in 1:10)
    {
      if(compare[p,2]==compare[p,3])
      {
        compare[p,4]=1
      }else
      {
        compare[p,4]=0
      } 
    }
    
    #Misclassification percent from the test data
    
    mce<-(1-(sum(compare[,4]))/10)*100
    
    all_mce[i]<-mce
    
    all_compare<-as.data.frame(rbind(all_compare, compare))
    
  }
  
  all_compare<-as.data.frame(all_compare[2:nrow(all_compare),])
  all_compare_add<-rep(NA,nrow(all_compare))
  all_compare<-as.data.frame(cbind(all_compare,all_compare_add))
  
  #All lambdas from 100 iterations
  all_lambda<-as.data.frame(all_lambda)
  colnames(all_lambda)<-c("Lambda")
  
  #All cross-validation misclassification errors for each best lambda
  all_cvm<-as.data.frame(all_cvm)
  colnames(all_cvm)<-c("cvm")
  
  #Misclassification percentage from the test data
  all_mce<-as.data.frame(all_mce)
  colnames(all_mce)<-c("mce")
  
  comb<-as.data.frame(cbind(all_lambda,all_cvm, all_mce))
  
  #Define false positives and false negatives from the test data
  
  for (i in 1:nrow(all_compare))
  {
    if(all_compare[i,2]=="Y" && all_compare[i,3]=="Y")
    {
      all_compare[i,5]="TP"
    }else if(all_compare[i,2]=="Y" && all_compare[i,3]=="N")
    {
      all_compare[i,5]="FN"
    }else if(all_compare[i,2]=="N" && all_compare[i,3]=="N")
    {
      all_compare[i,5]="TN"
    }else
    {
      all_compare[i,5]="FP"
    }
  }
  
  names(all_compare)[5] <- "Outcome"
  TP<-nrow(subset(all_compare, Outcome=="TP"))
  TN<-nrow(subset(all_compare, Outcome=="TN"))
  FN<-nrow(subset(all_compare, Outcome=="FN"))
  FP<-nrow(subset(all_compare, Outcome=="FP"))
  
  #Accuracy rate
  accuracy<-round((TP+TN)/(TP+FN+FP+TN)*100)
  
  all_accuracy[j]<-accuracy
  
  ##########################################################################################
  #Fit lasso on the full data set 
  ##########################################################################################
  
  out=glmnet(x,y, alpha=1, lambda=grid, family = "binomial")
  
  all_lambda2<- rep(NA, 100)
  all_cvm2<- rep(NA, 100)
  
  #Get lambda and mean misclassification error for each iteration
  
  for (i in 1:100)
  {
    cv.out=cv.glmnet(x,y,alpha=0.5,family = "binomial",type.measure = "class", nfolds=5)
    
    bestlam=cv.out$lambda.min
    
    lambda<-as.data.frame(cv.out$lambda)
    cvm<-as.data.frame(cv.out$cvm)
    
    index <- as.numeric(which(lambda==bestlam))
    
    error<-cvm[index,]
    
    all_lambda2[i]<-bestlam
    all_cvm2[i]<-error
  }
  
  #All lambdas from 100 iterations
  all_lambda2<-as.data.frame(all_lambda2)
  colnames(all_lambda2)<-c("Lambda")
  
  #All cross-validation misclassification errors for each best lambda
  all_cvm2<-as.data.frame(all_cvm2)
  colnames(all_cvm2)<-c("cvm")
  
  comb2<-as.data.frame(cbind(all_lambda2,all_cvm2))
  
  inds_all = as.data.frame(which(comb2$cvm == min(comb2$cvm), arr.ind=TRUE))
  colnames(inds_all)<-c("row")
  inds_all1<-comb2[row.names(comb2) %in% inds_all$row,]
  
  #Lambda value with the smallest mean misclassification error:
  lambda_wmincvm2<-inds_all1[1,1]
  
  lasso.coef=predict(out,type="coefficients", s=lambda_wmincvm2)
  lasso.coef2<-as.matrix(lasso.coef)
  lasso.coef3<-as.data.frame(lasso.coef2[2:nrow(lasso.coef2)])
  lasso<-as.data.frame(cbind(names(data)[2:138],lasso.coef3))
  colnames(lasso)<-c("Feature", "Value")
  InterestingFeatures<-subset(lasso, abs(Value)>0)
  
  bind<-cbind.data.frame(header, InterestingFeatures[, "Feature"][match(rownames(header), rownames(InterestingFeatures))])
  bind<-data.frame(bind[,2])
  bind[] <- lapply(bind, as.character)
  names(bind)<-"new"
  
  bind[!is.na(bind)] <- 1
  bind[is.na(bind)] <- 0
  
  bind$new<-as.numeric(bind$new)
  
  all_predictors<-all_predictors+bind
}

MostImportantFeatures<-cbind.data.frame(header,all_predictors)
write.table(MostImportantFeatures,"~/Dropbox/Berkeley_Projects/CS289/MostImportantFeatures.txt", quote = FALSE)

