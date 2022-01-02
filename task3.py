# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 15:13:43 2022

@author: yasmenkhaled
"""

import pandas as pd
import numpy as np
dataset = pd.read_csv("diabetes.csv")
print(dataset.head())


features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction","Age"]
X = dataset[features]
output = ["Outcome"]
y = dataset[output]
print(X.head())

print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.2)


# Clean data from missing values >>
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_test = pd.DataFrame(imputer.transform(X_test))

imputed_X_train.columns = X_train.columns
imputed_X_test.columns = X_test.columns



from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
model = DecisionTreeRegressor()
model.fit(imputed_X_train, y_train)
X_validation = model.predict(imputed_X_test)
decision_tree_error = mean_absolute_error( y_test ,X_validation)
print(decision_tree_error)


#Random forest
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=0)
forest_model.fit(imputed_X_train, y_train.values.ravel())
X_validation_forest = forest_model.predict(imputed_X_test)
random_forest_error = mean_absolute_error(y_test, X_validation_forest)
print(random_forest_error)


#Logistic 
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter= 180)
logistic_model.fit(imputed_X_train, y_train.values.ravel())
X_validation_logistic = logistic_model.predict(imputed_X_test)
logistic_error = mean_absolute_error(y_test, X_validation_logistic)
print(logistic_error)


 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

#feature scaling 
X_scale = StandardScaler()
imputed_X_train = X_scale.fit_transform(imputed_X_train)
imputed_X_test = X_scale.transform(imputed_X_test)




from scipy.sparse.sputils import matrix
KNN = KNeighborsClassifier(n_neighbors= 27, p=2, metric='euclidean')
KNN.fit(imputed_X_train, y_train.values.ravel())
X_validation_KNN = KNN.predict(imputed_X_test)



output = confusion_matrix(y_test, X_validation_KNN)
print(output)


print(f"the goal for KNN is {f1_score(y_test, X_validation_KNN)}")
print(f"the accur for KNN is {accuracy_score(y_test, X_validation_KNN)} ")
print(f"the goal for decision tree is {f1_score(y_test, X_validation)}")
print(f"the accur for decision tree is {accuracy_score(y_test, X_validation)} ")
print(f"the goal for logistic is {f1_score(y_test, X_validation_logistic)}")
print(f"the accur for logistic is {accuracy_score(y_test, X_validation_logistic)} ")
