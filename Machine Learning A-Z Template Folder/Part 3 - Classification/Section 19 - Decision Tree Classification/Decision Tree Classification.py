  #  DECISION TREE CLASSIFICATION

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Setting working directory #
#===========================#

# Using the os library
os.getcwd()
os.chdir('D:\\Work\\ML\\Super Data Science\\Machine Learning A-Z Template Folder\\Part 3 - Classification\\Section 19 - Decision Tree Classification\\Decision_Tree_Classification')


# Dataset #
#=========#

# Importing the dataset #
dataset = pd.read_csv('Social_Network_Ads.csv')

# Spliting data into dependent and independent variables
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values


# Spliting the data into training set and test set #
#==================================================#

# Using the train_test_split class from cross-validation library of sklearn #
#---------------------------------------------------------------------------#
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = .25, random_state=0)


# Feature Scaling #
#=================#

# Decision trees does not require feature scaling as the trees are not algorithms based on the Euclidean distance
# Thus feature scaling is not required.
# However in part of visualization the is a requirement in high resolution thus feature scaling would be useful as it would make process fast.


# Using the standard scaler class from the preprocessing library of sklearn #
#---------------------------------------------------------------------------#
from sklearn.preprocessing import StandardScaler
Sc_x = StandardScaler()
X_train = Sc_x.fit_transform(X_train)
X_test = Sc_x.fit_transform(X_test)

# Dependent variable is not require to be trained
'''
Sc_y = StandardScaler()
Y_train = Sc_y.fit_transform(Y_train)
Y_test = Sc_y.fit_transform(Y_test)
'''

# Classification Model #
#======================#

# Fitting the Decision tree to the Dataset
# Using the DecisionTreeClassifier class from the tree library of sklearn #
#---------------------------------------------------------------------------#
from sklearn.tree import DecisionTreeClassifier

# Creating an object of DecisionTreeClassifier class and this object will be the classifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# Making Decision tree based on entropy, by setting the criterion parametet to entropy
# It will make the splits more homogenious.

# Using the fit method
# The fit method is a tool of a function that will fit the regressor objectives into the trainning set.
# Fitting the regressor object to trainning set.
classifier.fit(X_train, Y_train)

# Prediction
# Using the predict method
# The predict method is a tool of function that will make the prediction of observation.
# Creating a vector of predicting values
Y_pred = classifier.predict(X_test)


# Evaluation of the classifier
# ---------------------------

# Making the confusion Matrix
# Confusion matrix evaluates the accuracy of a classification
# Using the confusion_matrix function from matrics library of sklearn
#---------------------------------------------------------------------#
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
# Take input of real value and the prdicted values


# Visualization #
#===============#

# Using the ListedColormap class from the colors library of matplotlib #
#----------------------------------------------------------------------#
from matplotlib.colors import ListedColormap

# Visualizing the Training set results
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() -1, stop = X_set[:,0].max() +1, step = 0.01),
                     np.arange(start = X_set[:,1].min() -1, stop = X_set[:,1].max() +1, step = 0.01))
# Using the meshgrid method of numpy
# The arange methord is a function that return coordinate matrices from coordinate vectors.
# Using the arange method of numpy
# The arange methord is a function that creates an array of incremented values

# The contourf function of matplot make contour betweem the two prediction region.
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
# Applying classifier to pridict on every pixel obsevation points
# The ListedColormap method is used to colorize the pixel point depending on the prediction output.

# Setting the plot limit
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Making a loop to plot all the datapoints
for i,j in enumerate (np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j,1],
                c = ListedColormap(('red','green'))(i),label = j)
plt.title('Decision Tree (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
   
# Visualizing the Test set results
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() -1, stop = X_set[:,0].max() +1, step = 0.01),
                     np.arange(start = X_set[:,1].min() -1, stop = X_set[:,1].max() +1, step = 0.01))
# Using the meshgrid method of numpy
# The arange methord is a function that return coordinate matrices from coordinate vectors.
# Using the arange method of numpy
# The arange methord is a function that creates an array of incremented values

# The contourf function of matplot make contour betweem the two prediction region.
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
# Applying classifier to pridict on every pixel obsevation points
# The ListedColormap method is used to colorize the pixel point depending on the prediction output.

# Setting the plot limit
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Making a loop to plot all the datapoints
for i,j in enumerate (np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j,1],
                c = ListedColormap(('red','green'))(i),label = j)
plt.title('Decision Tree (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()