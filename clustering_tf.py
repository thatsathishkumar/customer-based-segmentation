import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA  # Import PCA
import joblib

# Load the TensorFlow model
model_path = 'output/model.h5'
model = tf.keras.models.load_model(model_path)

# Generate sample data
X, _ = make_blobs(n_samples=278, centers=4, cluster_std=0.60, random_state=0)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)  # Specify the desired number of components
X_pca = pca.fit_transform(X)

# Use the model to predict cluster labels
cluster_labels = model.predict(X_pca)

# Plot the data points with cluster labels after dimensionality reduction
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.argmax(cluster_labels, axis=1), cmap='viridis', s=50, alpha=0.8)
plt.title('Clustering with TensorFlow Model (Dimensionality Reduction)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
