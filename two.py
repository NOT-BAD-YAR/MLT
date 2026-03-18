import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from xgboost import XGBClassifier 

# Load the dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 

# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Decision Tree Classifier 
dt = DecisionTreeClassifier(criterion='gini') 
dt.fit(X_train, y_train) 
y_pred_dt = dt.predict(X_test) 
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt)) 
print(confusion_matrix(y_test, y_pred_dt)) 
print(classification_report(y_test, y_pred_dt)) 

# Random forest Classifier 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train) 
y_pred_rf = rf.predict(X_test) 
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf)) 
print(confusion_matrix(y_test, y_pred_rf)) 
print(classification_report(y_test, y_pred_rf)) 

# XGBoost Classifier 
xgb = XGBClassifier(eval_metric='mlogloss') 
xgb.fit(X_train, y_train) 
y_pred_xgb = xgb.predict(X_test) 
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb)) 
print(confusion_matrix(y_test, y_pred_xgb)) 
print(classification_report(y_test, y_pred_xgb))
