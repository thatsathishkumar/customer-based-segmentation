
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Assuming you have loaded your data into X (features) and y (target)

# Split data into train and test sets
from testing_model import *

# Normalize feature data
X_train_normalized = X_train / 255.0  # Assuming pixel values are in the range [0, 255]
X_test_normalized = X_test / 255.0


# Convert DataFrame to NumPy array
X_train_array = X_train_normalized.values
X_test_array = X_test_normalized.values

# Define the fully connected neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=X_train_array.shape[1:]),  # Flatten input data
    tf.keras.layers.Dense(128, activation='relu'),  # Add a fully connected layer with 128 units
    tf.keras.layers.Dense(64, activation='relu'),  # Add another fully connected layer with 64 units
    tf.keras.layers.Dense(10)  # Example: 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits =True),
              metrics=['accuracy'])

# Train the model
start_time = time.time()
history = model.fit(X_train_array, y_train, epochs=30, validation_split=0.1)
end_time = time.time()

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test_array, y_test)

print("""
      TENSORFLOW
      ----------
      """)

print("Test Accuracy:", test_accuracy)

# Print time taken for training
print("Time taken for training: {:.2f} seconds".format(end_time - start_time))

# Save the model in HDF5 format
path = r'output/model.h5'
model.save(path, save_format='h5')

print("Model saved...")
