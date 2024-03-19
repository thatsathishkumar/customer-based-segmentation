import numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from sklearn.cluster import *
from sklearn.datasets import *


from testing_model import *

# Normalize feature data
X_train_normalized = X_train / 255.0  # Assuming pixel values are in the range [0, 255]
X_test_normalized = X_test / 255.0


# Convert DataFrame to NumPy array
X_train_array = X_train_normalized.values
X_test_array = X_test_normalized.values


# Load the saved model

path = r'output/model.h5'
loaded_model = tf.keras.models.load_model(path)

# Assuming you have your dataset
# X = ...

# Generate embeddings using the loaded model
embeddings = loaded_model.predict(X)

# Calculate WCSS for different values of k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(embeddings)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method with tensorflow')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(np.arange(1, 11, 1))
plt.grid(True)
plt.show()
