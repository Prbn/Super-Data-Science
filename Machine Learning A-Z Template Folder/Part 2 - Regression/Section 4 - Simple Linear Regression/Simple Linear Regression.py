# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset #
#=========#
# Importing the dataset #
salary_d = pd.read_csv('Salary_Data.csv')
x = salary_d.iloc[:,:-1].values
y = salary_d.iloc[:,1].values

# Spliting the data into training set and test set #
#==================================================#

# Using the cross-validation library of sklearn #
#-----------------------------------------------#
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 1/3, random_state=0)

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

###################

# Fitting simple linear regression model to the Training set #
#============================================================#
# Using the linear_model library from sklearn
# Using the linearRegression class from the linear model library
#------------------------------------------------------------------------#
from sklearn.linear_model import LinearRegression

# Creating an object of linear regression class and this object will be the regressor
regressor = LinearRegression()
# LinearRegression() is like a function that returns an object of it self

# Using the fit method
# The fit method is a tool of a function that will fit the regressor objectives into the trainning set.
# Fitting the regressor object to trainning set.
regressor.fit(X_train, Y_train)

# Predicting the Test Set results #
#=================================#

# Using the predict method
# The predict method is a tool of function that will make the prediction of observation.
regressor.predict(X_test)
# Creating a vector of predicting values
Y_pred = regressor.predict(X_test)

# Visualization #
#===============#

# Visualising the Training set result
# Using the pyplot library from matplotlib #
#------------------------------------------#

# The scatter meathod is a tool of function to make a scatter plot of the observation points.
plt.scatter(X_train,Y_train,color = 'red')
# Making the data point in plot red in color
 
# Ploting the regression line for the training data
# Using the plot method
# The plot method is a tool of function to make a linear plot of the observation point. 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

# Giving aesthetics
# Title of the plot using title method.
plt.title('Salary vs Experience(Trainning set)')
# X axis label of the plot using the xlabel method
plt.xlabel('Years of Experience')
# Y axis label of the plot using the ylabel method
plt.ylabel('Salary')

# Showing the graph
# Using the show method is a tool of function that compiles and shows all the plots.
# The show method 
plt.show()

# Visuaizing the test set results
plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Salary vs Experience(Trainning set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

###################
del salary_d,y

