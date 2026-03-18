import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

# Load the dataset
mnist = fetch_openml('mnist_784', version=1) 
X = mnist.data 
y = mnist.target.astype(int) 

# Normalize 
X = X / 255.0 
print("Dataset Loaded Successfully") 

# Train Test Split 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42 
) 

# SVM Model 
svm = SVC(kernel='rbf', gamma='scale') 
svm.fit(X_train, y_train) 
y_pred_svm = svm.predict(X_test) 
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm)) 
print(confusion_matrix(y_test, y_pred_svm)) 
print(classification_report(y_test, y_pred_svm)) 

# KNN Model 
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train) 
y_pred_knn = knn.predict(X_test) 
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn)) 
print(confusion_matrix(y_test, y_pred_knn)) 
print(classification_report(y_test, y_pred_knn)) 

# Visualization 
plt.figure(figsize=(8,4)) 
for i in range(10): 
    plt.subplot(2,5,i+1) 
    plt.imshow(X_test.iloc[i].values.reshape(28,28), cmap='gray') 
    plt.title("Predicted: " + str(y_pred_knn[i])) 
    plt.axis('off') 
plt.show()
