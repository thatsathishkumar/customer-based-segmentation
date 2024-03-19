import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

csv_path  = r"dataset\Samples.csv"
X = pd.read_csv(csv_path)
y = np.load(r"dataset\Target.npy")

# Now, perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size =  0.2)


