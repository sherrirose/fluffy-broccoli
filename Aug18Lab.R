##################################################################
##LASSO:Least Absolute Shrinkage and Selection Operator         ##
##Shrinks some coefficients to zero, provides variable selection##
##################################################################
set.seed(27);n<-1000
W1=runif(n, min = .5, max = 1)
W2=runif(n, min = 0, max = 1)
W3=runif(n, min = .25, max = .75)
W4=runif(n, min = 0, max = 1)
Y=rnorm(n, 4*W1+3*W2+W3, sd=.5)
W <- matrix(c(W1,W2,W3,W4), nrow=n, ncol=4)
colnames(W)<-c("W1","W2","W3","W4")

library(glmnet)
lasso1<-glmnet(W,Y)
plot(lasso1, label=TRUE)
lasso1

cv.lasso1<-cv.glmnet(W,Y)
plot(cv.lasso1)
coef(cv.lasso1, s="lambda.min")

head(predict(cv.lasso1, newx = W, s = "lambda.min"))

##################################################################
##RIDGE                                                         ##
##Shrinks coefficients, but none are exactly zero               ##
##################################################################

ridge1<-glmnet(W,Y, alpha=0)
plot(ridge1, label=TRUE)
ridge1

cv.ridge1<-cv.glmnet(W,Y,alpha=0)
plot(cv.ridge1)
coef(cv.ridge1, s="lambda.min")

##################################################################
##RPART                                                         ##
##Define splits based on homogeneity for the outcome            ##
##################################################################

library(rpart)

rpart1<-rpart(Y~W1+W2)
plot(rpart1)
text(rpart1, use.n=TRUE, cex=0.79)

head(predict(rpart1))

##################################################################
##ENSEMBLING!                                                   ##
##Take a weighted average of multiple algorithms                ##
##################################################################
library(SuperLearner)

set.seed(27);n<-500
data <- data.frame(W1=runif(n, min = .5, max = 1),
W2=runif(n, min = 0, max = 1),
W3=runif(n, min = .25, max = .75),
W4=runif(n, min = 0, max = 1))
data <- transform(data, #add W5 dependent on W2, W3
W5=rbinom(n, 1, 1/(1+exp(1.5*W2-W3))))
data <- transform(data, #add Y
Y=rbinom(n, 1,1/(1+exp(-(-.2*W5-2*W1+4*W5*W1-1.5*W2+sin(W4))))))

#Specify a library of algorithms
SL.library <- c("SL.glm", "SL.randomForest", "SL.glmnet")

#Run the super learner to obtain final predicted values for the super learner
#as well as CV risk for algorithms in the library
fit.data.SL<-SuperLearner(Y=data[,6],X=data[,1:5],
	SL.library=SL.library, family=binomial(),
	method="method.NNLS", verbose=TRUE)

#CV risks for algorithms in the library
fit.data.SL

#Run the cross-validated super learner to obtain its CV risk
fitSL.data.CV <- CV.SuperLearner(Y=data[,6],X=data[,1:5], V=10,
	SL.library=SL.library, verbose = TRUE,
	method = "method.NNLS", family = binomial())

#CV risk for super learner
mean((data[,6]-fitSL.data.CV$SL.predict)^2)
