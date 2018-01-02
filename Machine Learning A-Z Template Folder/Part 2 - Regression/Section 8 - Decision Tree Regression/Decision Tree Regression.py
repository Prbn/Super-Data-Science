# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Setting working directory #
#===========================#

# Using the os library
os.getcwd()
os.chdir('D:\\Work\\ML\\Super Data Science\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 8 - Decision Tree Regression\\Decision_Tree_Regression')


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
'''
# Using the standard scaler class from the preprocessing library of sklearn #
#---------------------------------------------------------------------------#
from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.fit_transform(X_test)
Sc_Y = StandardScaler()
Y_train = sc_y.fit_trandform(Y_train)
'''
# The library used for building the linear model already has the feature scaling built into it

# Regression Model #
#==================#

# Fitting the Random Forest Regreesion to the Dataset
# Using the RandomForestRegressor class from the tree library of sklearn #
#---------------------------------------------------------------------------#
from sklearn.tree import DecisonTreeRegressor

# Creating an object of Random Forest Regreesion class and this object will be the regressor
regressor = DecisonTreeRegressor(random_state = 0 )
# DecisonTreeRegressor() is like a function that returns an object of it self

# Using the fit method
# The fit method is a tool of a function that will fit the regressor objectives into the trainning set.
# Fitting the regressor object to trainning set.
regressor.fit(x, y)

# Prediction
# Using the predict method
# The predict method is a tool of function that will make the prediction of observation.
# Creating a vector of predicting values
Y_pred = regressor.predict(6.5)

# Visualization #
#===============#

# The decision tree regression require high resolution visualization
# Increasing the resolution
# Requires a variable containing better incrementation between levels
# Using the arange method of numpy
# The arange methor is a function that creates an array of incremented values
x_grid = np.arange(min(x),max(x),0.01)
# The method takes in minimum, maximum and increment steps
# The output is a vector
# Using reshape method to make it a matrix of number
x_grid = x_grid.reshape((len(x_grid),1))

# Plotting both the graph
# Replacing x by x_grid
plt.scatter(x,y,color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Salary vs Level(Decision Tree Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()