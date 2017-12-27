# Supoport Vector Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Setting working directory #
#===========================#

# Using the os library
os.getcwd()
os.chdir('D:\\Work\\ML\\Super Data Science\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 7 - Support Vector Regression (SVR)\\SVR')


# Dataset #
#=========#

# Importing the dataset #
dataset = pd.read_csv('Position_Salaries.csv')

# Spliting data into dependent and independent variables
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


# Spliting the data into training set and test set #
#==================================================#
# Doesnt require test set in this perticular excercise 
'''
# Using the cross-validation library of sklearn #
#-----------------------------------------------#
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 1/3, random_state=0)
'''

# Feature Scaling #
#=================#

# Using the standard scaler class from the preprocessing library of sklearn #
#---------------------------------------------------------------------------#
from sklearn.preprocessing import StandardScaler
Sc_x = StandardScaler()
x = Sc_x.fit_transform(x)

Sc_y = StandardScaler()
y = Sc_y.fit_transform(y)

# The library SVR does not have feature scaling built into it.

# Regression Model #
#==================#

# Fittting SVR to the dataset
# Using the SVR Class from the SVM library of sklearn
#-----------------------------------------------------#
from sklearn.svm import SVR
# Creating an object regressor
regressor = SVR(kernel = 'rbf')
# Specifies the kernal type to be used as gausian which is 'rbf'
# Fitting
# Using fit method
regressor.fit(x,y)
# SVR regressor has been created

# Prediction #
#============#

# Predicting a new result
regressor.predict(Sc_x.transform(np.array([[6.5]])))
# The new value must be transformed to scale
# Using the Sc_x or standard scaler of x and using the transform function
# The transform function take an array as an input so the value is to be converted into an array
# Using the np.array([[]]) function to convert the function into an array
# Requires an inverse scale of the output from prediction.
# Inverse of the standard scaler of y
# Using the inverse_transform() function
Sc_y.inverse_transform(regressor.predict(Sc_x.transform(np.array([[6.5]]))))

y_pred = Sc_y.inverse_transform(regressor.predict(Sc_x.transform(np.array([[6.5]]))))






# Visualization #
#===============#

# Visualising the SVR results
plt.scatter(x,y,color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Salary vs Level(Polynomial Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()













