import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import joblib

# Load the KMeans model
path = 'output/kmeans_model.pkl'
kmeans = joblib.load(path)

# Generate sample data
X, _ = make_blobs(n_samples=278, centers=4, cluster_std=0.60, random_state=0)

# Calculate the IQR for each feature
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1

# Define a threshold multiplier
threshold_multiplier = 1.5

# Define lower and upper bounds for outliers detection
lower_bound = Q1 - threshold_multiplier * IQR
upper_bound = Q3 + threshold_multiplier * IQR

# Identify outliers
outliers_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)

# Filter out outliers
X_filtered = X[~outliers_mask]

# Plot the data points and cluster centers after removing outliers
plt.figure(figsize=(8, 6))
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=kmeans.labels_[:len(X_filtered)], cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('KMeans Clustering (Outliers Removed)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()
