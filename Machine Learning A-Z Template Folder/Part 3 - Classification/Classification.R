

##################
# CLASSIFICATION #
##################

# Logistic Regression #
#=====================#

# Importing Data
#---------------

# Setting up new working directory using the
setwd('D:\\Work\\ML\\Super Data Science\\Machine Learning A-Z Template Folder\\Part 3 - Classification')

# Putting the data frame into an object called dataset
dataset <- read.csv('Section 14 - Logistic Regression\\Logistic_Regression\\Social_Network_Ads.csv')

# Exploring data
# --------------

# Summary of dataset
summary(dataset)
# Structure of dataset
str(dataset)

# Cleaning the dataset
# --------------------

# Only extracting the independent and dependent variables
dataset <- dataset[,3:5]

# Splitting the data set
# ----------------------
  
# Splitting the dataset into the training set and Test set
# Using the package caTools
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = .75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# ---------------

training_set[,1:2] <- scale(training_set[,1:2])
test_set[,1:2] <- scale(test_set[,1:2])

# Regression model
# ----------------

# Using the randomForest package
library(randomForest)

# Fitting Logistic Regression to the training_set
# Using the glm() function under the randomForest library
# creating classifier variable to store the model
classifier <- glm(formula = Purchased ~ .,family = binomial, data = training_set)
# The family arguement is set to binomialfor logistic regression 


# Info about the classifier using summary() function
summary(classifier)

# Prediction
# ----------
# Let Aa be the new result
Aa = 6.5

# Predicting a new result with logistic Regression
prob_pred = predict(classifier,type = 'response', newdata = test_set[-3])
# The type arguement is set to 'response' for logistic regression
# The response tyoe gives probabilities in single vector
# test_set[-3] to remove the dependent variable

# Creating a vector of predicted results
y_pred = ifelse(prob_pred >0.5,1,0)

# Evaluating the model
# Making the confusion Matrix
# Confusion matrix evaluates the accuracy of a classification
# Using the table() function
cm = table(test_set[,3], y_pred)

# Visualization
# -------------

# Visualising the training set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = training_set
# Getting high resolution of the data
# Using the seq() function to generate a set of vector with incrementation
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
# Forming a dataset
grid_set = expand.grid(X1,X2)
# Setting column names
colnames(grid_set) = c('Age', 'EstimatedSalary')
# Getting a vector set of probability of each datapoint
prob_set = predict(classifier, type ='response', newdata = grid_set)
# getting an array of sutable output 
y_grid = ifelse(prob_set > 0.5,1,0)
plot(set[,-3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

# Visualising the test set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
grid_set = expand.grid(X1,X2)

colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type ='response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5,1,0)
plot(set[,-3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

###
rm(dataset, classifier, prob_pred, prob_set,X1,X2, Aa,training_set,test_set,cm,split,y_grid,y_pred,set,grid_set)

#==============================================================================#

# K NEAREST NEIGHBORS #
#=====================#

# Importing Data
#---------------

# Setting up new working directory using the
setwd('D:\\Work\\ML\\Super Data Science\\Machine Learning A-Z Template Folder\\Part 3 - Classification')

# Putting the data frame into an object called dataset
dataset <- read.csv('Section 15 - K-Nearest Neighbors (K-NN)\\K_Nearest_Neighbors\\Social_Network_Ads.csv')

# Exploring data
# --------------

# Summary of dataset
summary(dataset)
# Structure of dataset
str(dataset)

# Cleaning the dataset
# --------------------

# Only extracting the independent and dependent variables
dataset <- dataset[,3:5]

# Splitting the data set
# ----------------------

# Splitting the dataset into the training set and Test set
# Using the package caTools
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = .75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# ---------------

training_set[,1:2] <- scale(training_set[,1:2])
test_set[,1:2] <- scale(test_set[,1:2])

# Classification model
# --------------------

# Using the class package
library(class)

# Fitting K Nearest Neighbors to the training_set and predicting the test_set result
# Here the two step of creating a classifier and predicting, is done all in once
y_pred = knn(train = training_set[,-3], test = test_set[,-3],
             cl = training_set[,3], k=5)
# Inputing trainning set as train
# Inputing test sets as test 
# Factor of true classifications in given through cl
# cl is the dependent variable
# k is the number of neighbor and is set to 5

# Evaluation
# ----------

# Evaluating the model
# Making the confusion Matrix
# Confusion matrix evaluates the accuracy of a classification
# Using the table() function
cm = table(test_set[,3], y_pred)

# Visualization
# -------------

# Visualising the training set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = training_set
# Getting high resolution of the data
# Using the seq() function to generate a set of vector with incrementation
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
# Forming a dataset
grid_set = expand.grid(X1,X2)
# Setting column names
colnames(grid_set) = c('Age', 'EstimatedSalary')
# Getting an array of sutable output 
y_grid = knn(train = training_set[,-3], test = grid_set[,-3],
             cl = training_set[,3], k=5)
# Plotting
plot(set[,-3],
     main = 'K Nearest Neighbors (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

# Visualising the test set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
grid_set = expand.grid(X1,X2)

colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = knn(train = training_set[,-3], test = grid_set[,-3],
             cl = training_set[,3], k=5)
plot(set[,-3],
     main = 'K Nearest Neighbors (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

###
rm(cm,dataset,grid_set,set,test_set, training_set, split, X1,X2,y_grid, y_pred)

#==============================================================================#

# Support Vector Machines and Kernal SVM #
#========================================#

# Importing Data
#---------------

# Setting up new working directory using the
setwd('D:\\Work\\ML\\Super Data Science\\Machine Learning A-Z Template Folder\\Part 3 - Classification')

# Putting the data frame into an object called dataset
dataset <- read.csv('Section 16 - Support Vector Machine (SVM)\\SVM\\Social_Network_Ads.csv')

# Exploring data
# --------------

# Summary of dataset
summary(dataset)
# Structure of dataset
str(dataset)

# Cleaning the dataset
# --------------------

# Only extracting the independent and dependent variables
dataset <- dataset[,3:5]

# Splitting the data set
# ----------------------

# Splitting the dataset into the training set and Test set
# Using the package caTools
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = .75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# ---------------

training_set[,1:2] <- scale(training_set[,1:2])
test_set[,1:2] <- scale(test_set[,1:2])

# Classification model
# --------------------

# Using the e1071 package
library(e1071)
# Two popular library for svm are e1071 and kernlab


# Fitting classifier to the training_set
# Using the svm() function under the e1071 library
# Creating classifier variable to store the model
classifier <- svm(formula = Purchased ~ .,type = 'C-classification', kernel ='radial', data = training_set)
# There are two type of SVM, svm for regression and svm for classification
# The type parameter is important and as it chooses the type of kernal to be used and is set to 'C-classification'.
# kernal is also an important parameter.
# Setting kernal as 'linear' will make svm classifier a linear classifier similar to logistic regression.
# Setting kernal as 'radial' for gaussian
# data is the data to be trained on

# Info about the classifier using summary() function
summary(classifier)

# Prediction
# ----------

# Creating a vector of predicted results
y_pred = predict(classifier, newdata = test_set[-3])

# Evaluating the model
# Making the confusion Matrix
# Confusion matrix evaluates the accuracy of a classification
# Using the table() function
cm = table(test_set[,3], y_pred)
cm

# Visualization
# -------------

# Visualising the training set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = training_set
# Getting high resolution of the data
# Using the seq() function to generate a set of vector with incrementation
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
# Forming a dataset
grid_set = expand.grid(X1,X2)
# Setting column names
colnames(grid_set) = c('Age', 'EstimatedSalary')
# Getting an array of sutable output 
y_grid = predict(classifier, newdata = grid_set)
plot(set[,-3],
     main = 'SVM Classification (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

# Visualising the test set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
grid_set = expand.grid(X1,X2)

colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, type ='response', newdata = grid_set)
plot(set[,-3],
     main = 'SVM Classification (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

###
rm(dataset, classifier, prob_pred, prob_set,X1,X2, Aa,training_set,test_set,cm,split,y_grid,y_pred,set,grid_set)

#==============================================================================#

# NAIVE BAYES CLASSIFIER #
#========================#

# Importing Data
#---------------

# Setting up new working directory using the
setwd('D:\\Work\\ML\\Super Data Science\\Machine Learning A-Z Template Folder\\Part 3 - Classification')

# Putting the data frame into an object called dataset
dataset <- read.csv('Section 18 - Naive Bayes\\Naive_Bayes\\Social_Network_Ads.csv')

# Exploring data
# --------------

# Summary of dataset
summary(dataset)
# Structure of dataset
str(dataset)

# Cleaning the dataset
# --------------------

# Only extracting the independent and dependent variables
dataset <- dataset[,3:5]


# Preprocessing
# -------------

# Factorizing the categorical variables
dataset$Purchased <- as.factor(dataset$Purchased)
# Encoding Categorical data
dataset$Purchased <- factor(dataset$Purchased,nlevels = levels(dataset$Purchased),labels = c(0:nlevels(dataset$Purchased)))
# Preprocessing and encoding of categorical data is requires in Naive Bayes classification in e1071 library


# Splitting the data set
# ----------------------

# Splitting the dataset into the training set and Test set
# Using the package caTools
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = .75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# ---------------

training_set[,1:2] <- scale(training_set[,1:2])
test_set[,1:2] <- scale(test_set[,1:2])

# Classification model
# --------------------

# Using the e1071 package
library(e1071)
# The library is also used for svm
# The naiveBayes requires factorizing and encoding of categorical data. IMPORTANT


# Fitting classifier to the training_set
# Using the naiveBayes() function under the e1071 library
# Creating classifier variable to store the model
classifier <- naiveBayes(x = training_set[-3], y = training_set$Purchased)
# Inputing the independent variable on x
# Inputing the dependent variable on y

# Info about the classifier using summary() function
summary(classifier)

# Prediction
# ----------

# Creating a vector of predicted results
y_pred = predict(classifier, newdata = test_set[-3])

# Evaluating the model
# Making the confusion Matrix
# Confusion matrix evaluates the accuracy of a classification
# Using the table() function
cm = table(test_set[,3], y_pred)
cm

# Visualization
# -------------

# Visualising the training set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = training_set
# Getting high resolution of the data
# Using the seq() function to generate a set of vector with incrementation
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
# Forming a dataset
grid_set = expand.grid(X1,X2)
# Setting column names
colnames(grid_set) = c('Age', 'EstimatedSalary')
# Getting an array of sutable output 
y_grid = predict(classifier, newdata = grid_set)
plot(set[,-3],
     main = 'Naive Bayes Classification (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

# Visualising the test set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.05)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.05)
grid_set = expand.grid(X1,X2)

colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[,-3],
     main = 'Naive Bayes (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

###
rm(dataset, classifier, prob_pred, prob_set,X1,X2, Aa,training_set,test_set,cm,split,y_grid,y_pred,set,grid_set)

#==============================================================================#


# DECISION TREE CLASSIFICATION #
#==============================#

# Importing Data
#---------------

# Setting up new working directory using the
setwd('D:\\Work\\ML\\Super Data Science\\Machine Learning A-Z Template Folder\\Part 3 - Classification')

# Putting the data frame into an object called dataset
dataset <- read.csv('Section 19 - Decision Tree Classification\\Decision_Tree_Classification\\Social_Network_Ads.csv')

# Exploring data
# --------------

# Summary of dataset
summary(dataset)
# Structure of dataset
str(dataset)

# Cleaning the dataset
# --------------------

# Only extracting the independent and dependent variables
dataset <- dataset[,3:5]


# Preprocessing
# -------------

# Factorizing the categorical variables
dataset$Purchased <- as.factor(dataset$Purchased)
# Encoding Categorical data
dataset$Purchased <- factor(dataset$Purchased,nlevels = levels(dataset$Purchased),labels = c(0:nlevels(dataset$Purchased)))
# Preprocessing and encoding of categorical data is requires in Naive Bayes classification in e1071 library


# Splitting the data set
# ----------------------

# Splitting the dataset into the training set and Test set
# Using the package caTools
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = .75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# ---------------

# The decision tree model is not based on Euclidean Distance.
# But in visualization, scaling makes it easy and efficient
training_set[,1:2] <- scale(training_set[,1:2])
test_set[,1:2] <- scale(test_set[,1:2])

# Classification model
# --------------------

# Using the rpart package
library(rpart)

# Fitting classifier to the training_set
# Using the rpart() function under the rpart library
# Creating classifier variable to store the model
classifier <- rpart(formula = Purchased ~ ., data = training_set)
# Inputing the independent variable on x
# Inputing the dependent variable on y

# Info about the classifier using summary() function
summary(classifier)

# Prediction
# ----------

# Creating a vector of predicted results
y_pred = predict(classifier, newdata = test_set[-3], type = 'class')
# The formula is dependent variable expressed as a linear combination of the independent variable
# The data on which the to train needs to be specified.
# The decision tree prediction is in matrix of probabilities of differt class
# type = 'class' is used to get a single array of prediction

# Evaluating the model
# Making the confusion Matrix
# Confusion matrix evaluates the accuracy of a classification
# Using the table() function
cm = table(test_set[,3], y_pred)
cm

# Visualization
# -------------

# Visualising the training set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = training_set
# Getting high resolution of the data
# Using the seq() function to generate a set of vector with incrementation
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
# Forming a dataset
grid_set = expand.grid(X1,X2)
# Setting column names
colnames(grid_set) = c('Age', 'EstimatedSalary')
# Getting an array of sutable output 
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[,-3],
     main = 'Decision Tree Classification (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

# Visualising the test set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.05)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.05)
grid_set = expand.grid(X1,X2)

colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[,-3],
     main = 'Decision Tree (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

# Plotting Decision tree
plot(classifier)
text(classifier)


###
rm(dataset, classifier, prob_pred, prob_set,X1,X2, Aa,training_set,test_set,cm,split,y_grid,y_pred,set,grid_set)

#==============================================================================#

# RANDOM FOREST CLASSIFICATION #
#==============================#

# Importing Data
# --------------

# Setting up new working directory using the
setwd('D:\\Work\\ML\\Super Data Science\\Machine Learning A-Z Template Folder\\Part 3 - Classification')

# Putting the data frame into an object called dataset
dataset <- read.csv('Section 20 - Random Forest Classification\\Random_Forest_Classification\\Social_Network_Ads.csv')

# Exploring data
# --------------

# Summary of dataset
summary(dataset)
# Structure of dataset
str(dataset)

# Cleaning the dataset
# --------------------

# Only extracting the independent and dependent variables
dataset <- dataset[,3:5]


# Preprocessing
# -------------

# Factorizing the categorical variables
dataset$Purchased <- as.factor(dataset$Purchased)
# Encoding Categorical data
dataset$Purchased <- factor(dataset$Purchased,nlevels = levels(dataset$Purchased),labels = c(0:nlevels(dataset$Purchased)))
# Preprocessing and encoding of categorical data is requires in Naive Bayes classification in e1071 library


# Splitting the data set
# ----------------------

# Splitting the dataset into the training set and Test set
# Using the package caTools
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = .75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# ---------------

# The decision tree model is not based on Euclidean Distance.
# But in visualization, scaling makes it easy and efficient
training_set[,1:2] <- scale(training_set[,1:2])
test_set[,1:2] <- scale(test_set[,1:2])

# Classification model
# --------------------

# Using the randomForest package
library(randomForest)

# Fitting classifier to the training_set
# Using the randomForest() function under the randomForest library
# Creating classifier variable to store the model
classifier <- randomForest(x = training_set[-3], y = training_set$Purchased, ntree = 10)
# Inputing the independent variable on x
# Inputing the dependent variable on y
# The ntrees is the number of trees to be made


# Info about the classifier using summary() function
summary(classifier)

# Prediction
# ----------

# Creating a vector of predicted results
y_pred = predict(classifier, newdata = test_set[-3], type = 'class')
# The decision tree prediction is in matrix of probabilities of differt class
# type = 'class' is used to get a single array of prediction

# Evaluating the model
# Making the confusion Matrix
# Confusion matrix evaluates the accuracy of a classification
# Using the table() function
cm = table(test_set[,3], y_pred)
cm

# Visualization
# -------------

# Visualising the training set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = training_set
# Getting high resolution of the data
# Using the seq() function to generate a set of vector with incrementation
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
# Forming a dataset
grid_set = expand.grid(X1,X2)
# Setting column names
colnames(grid_set) = c('Age', 'EstimatedSalary')
# Getting an array of sutable output 
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[,-3],
     main = 'Ranfom Forest (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

# Visualising the test set result

# Using the ElemStatLearn library
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.05)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.05)
grid_set = expand.grid(X1,X2)

colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[,-3],
     main = 'Random Forest (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set,pch = 21, bg = ifelse(set[,3]==1,'green4','red3'))

# Plotting Decision tree
plot(classifier)
text(classifier)


###
rm(dataset, classifier, prob_pred, prob_set,X1,X2, Aa,training_set,test_set,cm,split,y_grid,y_pred,set,grid_set)

#==============================================================================#









