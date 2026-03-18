# Import libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN 
from sklearn.preprocessing import StandardScaler 
from scipy.cluster.hierarchy import dendrogram, linkage 
 
# Load dataset 
data = pd.read_csv("Mall_Customers.csv") 

# Select important features 
X = data[['Annual Income (k$)', 'Spending Score (1-100)']] 
 
# Feature Scaling 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
 
# K-MEANS CLUSTERING 
kmeans = KMeans(n_clusters=5, random_state=0) 
kmeans_labels = kmeans.fit_predict(X_scaled) 
 
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans_labels, cmap='rainbow') 
plt.title("Customer Segmentation using K-Means") 
plt.xlabel("Annual Income") 
plt.ylabel("Spending Score") 
plt.show() 
 
# HIERARCHICAL CLUSTERING 
linked = linkage(X_scaled, method='ward') 
plt.figure(figsize=(8,5)) 
dendrogram(linked) 
plt.title("Dendrogram for Hierarchical Clustering") 
plt.xlabel("Customers") 
plt.ylabel("Distance") 
plt.show() 
 
hc = AgglomerativeClustering(n_clusters=5) 
hc_labels = hc.fit_predict(X_scaled) 
 
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=hc_labels, cmap='rainbow') 
plt.title("Customer Segmentation using Hierarchical Clustering") 
plt.xlabel("Annual Income") 
plt.ylabel("Spending Score") 
plt.show() 
 
# DBSCAN CLUSTERING 
dbscan = DBSCAN(eps=0.5, min_samples=5) 
db_labels = dbscan.fit_predict(X_scaled) 
 
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=db_labels, cmap='plasma') 
plt.title("Customer Segmentation using DBSCAN") 
plt.xlabel("Annual Income") 
plt.ylabel("Spending Score") 
plt.show() 
 
# Print cluster labels 
print("K-Means Clusters:\n", kmeans_labels) 
print("Hierarchical Clusters:\n", hc_labels) 
print("DBSCAN Clusters:\n", db_labels)
