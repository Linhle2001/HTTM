# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 23:11:59 2022

@author: Admin
"""
#cleaning data



#import numpy as np
import pandas as pd
df = pd.read_csv("BMX_G.csv")
print(df.shape)
# (9338, 27)  9338 rows and 27 columns
print(df.isnull().sum());
#BMXWAIST: Waist Circumference (cm) (chu vi vong eo)
#BMXLEG: Upper Leg Length (cm) (chu vi vong dui)
#df = df.dropna(subset=['bmxleg','bmxwaist']) # remove rows with NaNs
#print(df.shape)
# (6899, 27)




#splitting data

'''
import numpy as np;
from sklearn.model_selection import train_test_split;

x = np.arange(1, 25).reshape(12, 2);
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]);

x_train, x_test, y_train, y_test = train_test_split(x, y);
print("x_train");
print(x_train);
print("x_test");
print(x_test);
print("y_train");
print(y_train);
print("y_test");
print(y_test);
#default; 75-25%
'''
'''
import numpy as np;
from sklearn.model_selection import train_test_split;

x = np.arange(1, 25).reshape(12, 2);
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]);

x_train, x_test, y_train, y_test = train_test_split(x, y,\
                            test_size=4, random_state=4);
print("x_train");
print(x_train);
print("x_test");
print(x_test);
print("y_train");
print(y_train);
print("y_test");
print(y_test);
'''
'''
from sklearn.linear_model import LinearRegression;
from sklearn.model_selection import train_test_split;
import numpy as np;

x = np.arange(20).reshape(-1, 1);
y = np.array([5, 12, 11, 19, 30, 29, 23, 40, 51, 54, 74,\
              62, 68, 73, 89, 84, 89, 101, 99, 106]);
x_train, x_test, y_train, y_test = train_test_split(\
                            x, y, test_size=8, random_state=0);
model = LinearRegression().fit(x_train, y_train);
model.score(x_train, y_train);
model.score(x_test, y_test);
'''

'''
#Train-Test Split for Classification
# train-test split evaluation random forest on the sonar dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# fit the model
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
acc = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % acc)
'''

'''
#Train-Test Split for Regression
#train-test split evaluation random forest on the housing dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# fit the model
model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
'''

# import packages
'''
import pandas as pd
from sklearn.model_selection import train_test_split

# importing data
df = pd.read_csv('headbrain1.csv')
print(df.shape)

# head of the data
print(df.head())

X= df['Head Size(cm^3)']
y=df['Brain Weight(grams)']

# using the train test split function
X_train, X_test, y_train,\
y_test = train_test_split(X,y ,
      random_state=104,
      train_size=0.8, shuffle=True)

# printing out train and test sets
print('X_train : ')
print(X_train.head())
print(X_train.shape)
print('X_test : ')
print(X_test.head())
print(X_test.shape)
print('')
print('y_train : ')
print(y_train.head())
print(y_train.shape)
print('y_test : ')
print(y_test.head())
print(y_test.shape)
'''
