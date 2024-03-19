from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

# Assuming you have loaded your data into X (features) and y (target)
from testing_model import *

# Create an instance of KMeans
kmeans = KMeans(n_clusters = 4, n_init = 10)  # Example: 3 clusters

# Fit the KMeans model to the training data
start_time = time.time()
kmeans.fit(X_train)
end_time = time.time()

# Get cluster labels for both train and test data
train_cluster_labels = kmeans.predict(X_train)
test_cluster_labels = kmeans.predict(X_test)

# You can use these cluster labels for segmentation analysis

# Evaluate accuracy using a simple approach: mapping clusters to majority class
# Assuming we know the true class labels for train data
cluster_majority_class = [np.bincount(y_train[train_cluster_labels == i]).argmax() for i in tqdm(range(kmeans.n_clusters))]

# Map cluster labels to majority class
train_cluster_majority_pred = [cluster_majority_class[label] for label in tqdm(train_cluster_labels)]
test_cluster_majority_pred = [cluster_majority_class[label] for label in tqdm(test_cluster_labels)]

# Calculate accuracy
train_accuracy = accuracy_score(y_train, train_cluster_majority_pred)
test_accuracy = accuracy_score(y_test, test_cluster_majority_pred)
print("""
      K MEANS
      -------
      """)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Calculate and print time taken for training
print("Time taken for training: {:.2f} seconds".format(end_time - start_time))

import joblib
from sklearn.cluster import KMeans

# Assuming you have already trained your KMeans model
# kmeans = ...

# Define the file path where you want to save the model
path = 'output/kmeans_model.pkl'

# Save the model
joblib.dump(kmeans, path)