# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset #
#=========#

# Importing the dataset #
dataset = pd.read_csv('50_Startups.csv')

# Spliting data into dependent and independent variables
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encoding Categorical Values and Dummy Encoding
# Using siket learn LableEncoder, OneHotEncoder #
#-----------------------------------------------#
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])
# Here 3 signifies the column of the dataset which is to be encoded

# Using OneHotEncoder to make saperate column vector
onehotencoder = OneHotEncoder(categorical_features=[3])
# categorical_features = 3 signifies that the column number which is categorical
# Fitting to matrix x
x = onehotencoder.fit_transform(x).toarray()

# Avoiding dummy variable Trap
x = x[:,1:]
# The library used for building the linear model already has a built function to avoid dummy variable trap


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

# Multiple Linear Regression #
#============================#

# Fitting Multiple Linear Regression to the Training set
# Using the LinearRegression class from the lineat_model library of sklearn #
#---------------------------------------------------------------------------#
from sklearn.linear_model import LinearRegression

# Creating an object of linear regression class and this object will be the regressor
regressor = LinearRegression()
# LinearRegression() is like a function that returns an object of it self

# Using the fit method
# The fit method is a tool of a function that will fit the regressor objectives into the trainning set.
# Fitting the regressor object to trainning set.
regressor.fit(X_train, Y_train)

# Using the predict method
# The predict method is a tool of function that will make the prediction of observation.
regressor.predict(X_test)
# Creating a vector of predicting values
Y_pred = regressor.predict(X_test)

# Optimal Linear Regression Model #
#=================================#
# The main goal is to find the optimal combination of independent variables so that each independent variable of the team has great impact on the dependable variable.
# Each independent variable of the team is a powerful predictor and has a highly statistically significance.

# Using Backward Elimination

# Preparation 
# Using the statsmodel.formula.api library
#------------------------------------------#
import statsmodels.formula.api as sm

# The statsmodels does not take into account the constant or the y-intercept
# Therefore some modifications is required
# It requires an addition of a  matrix of independent variables

# Adding a column of 1 to the matrix of features
# Making an array of 1
# Using the ones function of numpy
np.ones((len(x),1)).astype(int)
# The astype(int) is used to make the type integer

# adding the column of 1
# Using the append function
x = np.append(arr = np.ones((len(x),1)).astype(int), values = x, axis = 1)
# First parameter is the array, it is the array in which the column is to be added
# The values is the column of values to be added
# The axis is used to denote the addition to column or to the rows

# Backward elimination
# Creating a new matrix of features which would be the optimal matrix of features
x_opt = x[:,[0,1,2,3,4,5]]
# Initailizing with all the independent variables
# It is done as in backward elimination starts with all the independent variables
# The indpendent variables that are statically insignificant are removed one by one

# Fitting the linear model with all possible predictors
# Creating a new regressor from statsmodel library
# The regressor will be an object of a new class of statsmodel library
# Using the ols class of the statsmodel library
regressor_ols = sm.OLS(endog = y, exog =x_opt).fit()
# endog is the independent variable input
# exog is the dependent variable input
# .fit() is done to fit the regressor model.

# Checking the p value
# Using the summary method of statsmodel library
# The summary function will return a table containing all the statistical metrics
regressor_ols.summary()
# lower the p value more significant the independent variable is to the dependable variable

# Removing the independent variable whos p value is greater than significant level
x_opt = x[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog =x_opt).fit()
regressor_ols.summary()

# Removing the independent variable whos p value is greater than significant level
x_opt = x[:,[0,3,5]]
regressor_ols = sm.OLS(endog = y, exog =x_opt).fit()
regressor_ols.summary()

# Removing the independent variable whos p value is greater than significant level
x_opt = x[:,[0,3]]
regressor_ols = sm.OLS(endog = y, exog =x_opt).fit()
regressor_ols.summary()










###################
