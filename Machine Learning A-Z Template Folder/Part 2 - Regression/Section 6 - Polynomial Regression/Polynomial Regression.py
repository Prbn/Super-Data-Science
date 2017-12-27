# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Setting working directory #
#===========================#

# Using the os library
os.getcwd()
os.chdir('D:\\Work\\ML\\Super Data Science\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Polynomial_Regression')


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
# The library used for building the model already has the feature scaling built into it

# Linear Regression model #
#=========================#

# The Linear Regression model would be used as a reference for comparision

# Fitting simple linear regression model to the dataset #
# Using the linear_model library from sklearn
# Using the linearRegression class from the linear model library
#----------------------------------------------------------------#
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


# Polynomial Regression Model #
#=============================#

# Including a new class to import some polynomial creating tools
# Using the PolynomialFeature class from the preprocessing library of sklearn #
#---------------------------------------------------------------------------#
from sklearn.preprocessing import PolynomialFeatures
# Using the poly_reg object
# The poly_reg is a transformet tool used to transform a matrix of features into a new matrix of features that contain polynomial features
poly_reg = PolynomialFeatures(degree=2)
# The degree is set to 2 as it is the degree of power of the polynomial
# Using the fit_transform method
x_poly=poly_reg.fit_transform(x)

# Fitting linear regression model to the Polynomial dataset #
# Using the linear_model library from sklearn
# Using the linearRegression class from the linear model library
#----------------------------------------------------------------#
from sklearn.linear_model import LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# Visualization #
#===============#

# Visualising the Simple linear model
# Using the pyplot library from matplotlib #
#------------------------------------------#
# Ploting employees Level by the number of salaries
# Using scatter method
# The scatter meathod is a tool of function to make a scatter plot of the observation points.
plt.scatter(x,y,color = 'red')
# Making the data point in plot red in color

# Ploting the Simple Regression line for the dataset
# Using the plot method
# The plot method is a tool of function to make a linear plot of the observation point. 
plt.plot(x, lin_reg.predict(x), color = 'blue')


# Giving aesthetics
# Title of the plot using title method.
plt.title('Salary vs Level(Simple Linear Regression)')
# X axis label of the plot using the xlabel method
plt.xlabel('Level')
# Y axis label of the plot using the ylabel method
plt.ylabel('Salary')

# Showing the graph
# Using the show method is a tool of function that compiles and shows all the plots.
# The show method 
plt.show()

# Plotting the Polynomial linear model
plt.scatter(x,y,color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'Green')
plt.title('Salary vs Level(Polynomial Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Plotting both the graph
plt.scatter(x,y,color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'Green')
plt.title('Salary vs Level(Polynomial Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Increasing degree
poly_reg_x = PolynomialFeatures(degree=4)
x_poly_x=poly_reg_x.fit_transform(x)
lin_reg_x = LinearRegression()
lin_reg_x.fit(x_poly_x,y)

# Plotting both the graph
plt.scatter(x,y,color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.plot(x, lin_reg_x.predict(poly_reg_x.fit_transform(x)), color = 'Green')
plt.title('Salary vs Level(Polynomial Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# For better visualization
# Increasing the resulution
# Requires a variable containing better incrementation between levels
# Using the arange method of numpy
# The arange methor is a function that creates an array of incremented values
x_grid = np.arange(min(x),max(x),0.1)
# The method takes in minimum, maximum and increment steps
# The output is a vector
# Using reshape method to make it a matrix of number
x_grid = x_grid.reshape((len(x_grid),1))

# Plotting both the graph
# Replacing x by x_grid
plt.scatter(x,y,color = 'red')
plt.plot(x_grid, lin_reg.predict(x_grid), color = 'blue')
plt.plot(x_grid, lin_reg_x.predict(poly_reg_x.fit_transform(x_grid)), color = 'Green')
plt.title('Salary vs Level(Polynomial Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Prediction #
#============#
# Prediction of new result with Simple linear Regression Model
lin_reg.predict(6.5)

# Prediction of a new result with polynomial linear Regression Model of degree 2
lin_reg_2.predict(poly_reg.fit_transform(6.5))

# Prediction of a new result with polynomial linear Regression Model of degree 4
lin_reg_x.predict(poly_reg_x.fit_transform(6.5))
















