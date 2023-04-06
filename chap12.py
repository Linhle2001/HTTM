# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:27:18 2022

@author: Admin
"""

import pandas as pd
import numpy as np
df = pd.read_csv("diabetes.csv")

#df.info()
#---check for null values---
print("Nulls")
print("=====")
print(df.isnull().sum())
#---check for 0s---
print("0s")
print("==")
print(df.eq(0).sum())
df[['Glucose','BloodPressure','SkinThickness',
 'Insulin','BMI','DiabetesPedigreeFunction','Age']] = \
 df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace
(0,np.NaN)

df.fillna(df.mean(), inplace = True)
'''
corr = df.corr()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)
ax.set_xticklabels(df.columns)
plt.xticks(rotation = 90)
ax.set_yticklabels(df.columns)
ax.set_yticks(ticks)
#---print the correlation factor---
for i in range(df.shape[1]):
    for j in range(9):
        text = ax.text(j, i, round(corr.iloc[i][j],2),
        ha="center", va="center", color="w")
plt.show()
'''
df = pd.read_csv('diabetes.csv')
#from sklearn import linear_model
from sklearn.model_selection import cross_val_score
#---features---
X = df[['Glucose','BMI','Age']]
#---label---
y = df.iloc[:,8]
#---number of folds---
folds = 10
#---creating odd list of K for KNN---
ks = list(range(1,int(len(X) * ((folds - 1)/folds)), 2))
#---perform k-fold cross validation---
from sklearn.neighbors import KNeighborsClassifier
#---empty list that will hold cv (cross-validates) scores---
cv_scores = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=folds, scoring='accuracy').mean()
    cv_scores.append(score)
#---get the maximum score---
knn_score = max(cv_scores)
#---find the optimal k that gives the highest score---
optimal_k = ks[cv_scores.index(knn_score)]
print(f"The optimal number of neighbors is {optimal_k}")
print(knn_score)

knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X, y)
'''

import pickle
#---save the model to disk---
filename = 'diabetes.sav'
#---write to the file using write and binary mode---
pickle.dump(knn, open(filename, 'wb'))


import pickle
from flask import Flask, request, jsonify
import numpy as np
app = Flask(__name__)
#---the filename of the saved model---
filename = 'diabetes.sav'
#---load the saved model---
loaded_model = pickle.load(open(filename, 'rb'))
@app.route('/diabetes/v1/predict', methods=['GET'])
def predict():
 #---get the features to predict---
 features = request.json
 #---create the features list for prediction---
 features_list = [features["Glucose"],
 features["BMI"],
 features["Age"]]
 #---get the prediction class---
 prediction = loaded_model.predict([features_list])
 #---get the prediction probabilities---
 confidence = loaded_model.predict_proba([features_list])
 #---formulate the response to return to client---
 response = {}
 response['prediction'] = int(prediction[0])
 response['confidence'] = str(round(np.amax(confidence[0]) * 100 ,2))
 return jsonify(response)
if __name__ == '__main__':
 app.run(host='0.0.0.0', port=5000)

 
'''
 # Creating the Client Application to Use the Model

import json
import requests
def predict_diabetes(BMI, Age, Glucose):
 url = 'http://127.0.0.1:5000/diabetes/v1/predict'
 data = {"BMI":BMI, "Age":Age, "Glucose":Glucose}
 data_json = json.dumps(data)
 headers = {'Content-type':'application/json'}
 response = requests.post(url, data=data_json, headers=headers)
 result = json.loads(response.text)
 return result
if __name__ == "__main__":
 predictions = predict_diabetes(30,40,100)
 print("Diabetic" if predictions["prediction"] == 1 else "Not Diabetic")
 print("Confidence: " + predictions["confidence"] + "%")
 #splitting data:
 '''    
import pandas as pd
from sklearn.model_selection import train_test_split

# importing data
df = pd.read_csv('diabetes.csv')
print(df.shape)

# head of the data
print(df.head())

X= df['Glucose']

y=df['Outcome']

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


from sklearn import linear_model
from sklearn.model_selection import cross_val_score
#---features---
X = df[['Glucose','BMI','Age']]
#---label---
y = df.iloc[:,8]
# evaluate Logistic Regression
log_regress = linear_model.LogisticRegression()
log_regress_score = cross_val_score(log_regress, X, y, cv=10, 
scoring='accuracy').mean()
print(log_regress_score)
#save result of algorithms
result = []
# append scoring using accuracy metrics
result.append(log_regress_score)

# evalueta K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
#---empty list that will hold cv (cross-validates) scores---
cv_scores = []
#---number of folds---
folds = 10
#---creating odd list of K for KNN---
ks = list(range(1,int(len(X) * ((folds - 1)/folds)), 2))
#---perform k-fold cross validation---
for k in ks:
 knn = KNeighborsClassifier(n_neighbors=k)# althorithms
 score = cross_val_score(knn, X, y, cv=folds, scoring='accuracy').mean()
 cv_scores.append(score)
#---get the maximum score---
knn_score = max(cv_scores)
#---find the optimal k that gives the highest score---
optimal_k = ks[cv_scores.index(knn_score)]
print(f"The optimal number of neighbors is {optimal_k}")
print(knn_score)
result.append(knn_score)

#evaluate Support Vector Machines
from sklearn import svm
linear_svm = svm.SVC(kernel='linear')
linear_svm_score = cross_val_score(linear_svm, X, y,
 cv=10, scoring='accuracy').mean()
print(linear_svm_score)
result.append(linear_svm_score)
'''