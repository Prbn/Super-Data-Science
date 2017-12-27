# Data Prepocessing

# Importing the libraries #
#=========================#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset #
#=========#
# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Matrix of fetures
# Creating a matrix of features.
# ':' is used to denote all values
X = dataset.iloc[:,:-1].values
# [:,:-1] all value of rows and all values of column except the last
Y = dataset.iloc[:,3].values
# [:,3] all values of rows of 3rd column

# Adding a column
# Using the append() function of numpy
X = np.append(arr = X, values = X, axis = 1 )
# First parameter is the array, it is the array in which the column is to be added
# The values is the column of values to be added.
# The axis is used to denote the addition to column or to the rows


# Dealing with a missing data #
#=============================#

# One missing value in age and another missing value in salary
# The missing age value is imputed using mean value

# Using siket learn Imputer #
#---------------------------#
from sklearn.preprocessing import Imputer
# sklearn is siket learn and contains libraries to do machine learning and has machine learning models
# Inside sklearn there is preprocessing library that contains a lot of classes method to preprocess
# Importing the Imputer class from preprocessing library of sklearn
# For age column
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# The missing value is set to the value which needs to be replaced, in this case it is NA
# The strategy is to find the median
imputer = imputer.fit(X[:,1:2])
X[:,1:2] = imputer.transform(X[:,1:2])
# For salary column
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X[:,2:3])
X[:,2:3] = imputer.transform(X[:,2:3])


# For all
"""
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
"""
#########################################################

# Encoding Categorical data #
#===========================#
# Using LableEncoder of siketlearn library #
#------------------------------------------#
from sklearn.preprocessing import LabelEncoder
# Label encoder is a class from scaler and pre-processing library
# Creating an object of label encoder class
labelencoder_X = LabelEncoder()
# Taking the fit_transform methord from label encoder

X[:,0] = labelencoder_X.fit_transform(X[:,0])
# fitting the label encode for x object to the first column country of our matrix X and return the first column country encoded

# Dummy Encoding
# Using siket learn LableEncoder, OneHotEncoder #
#-----------------------------------------------#
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features=[0])
# categorical_features = 0 signifies that the column number which is categorical in nature. 

# Fitting to matrix X
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Output Variable Y
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# Spliting the data into training set and test set #
#==================================================#

# Using the cross-validation library of sklearn #
#-----------------------------------------------#
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)


# Feature Scaling #
#=================#

# Using the standard scaler class from the preprocessing library of sklearn #
#---------------------------------------------------------------------------#
from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.fit_transform(X_test)

##########################################







































