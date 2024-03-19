import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import *
import tensorflow as tf

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


# Predict classes for the test set
y_pred = np.argmax(loaded_model.predict(X_test_array), axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=range(3), yticklabels=range(3))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
