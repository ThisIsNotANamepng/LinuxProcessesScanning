# Accuracy: 0.8849491477012634

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the CSV file into a DataFrame
df = pd.read_csv('../CombinedSets.csv')

cols_to_drop = [
    'ts',
    'EXC',
    'type',
    'PID'
]

## psutil can't get POLI, RTPR, TSLPI, or TSLPU but you can get it another way (see chatgpt)

df.drop(columns=cols_to_drop, inplace=True)

print("Columns:", df.columns.tolist())

# Split data into features and target variable
X = df.iloc[:, :-1]  # Features (all columns except the last one)
y = df.iloc[:, -1]   # Target variable (last column)

print(X.head())

le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

X = X.to_numpy().astype(np.float32)
y = y.to_numpy().astype(np.float32)

# Split the data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Reshape the data for CNN input (assuming 2D data)
# Here, you need to reshape it according to your data dimensions
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the CNN model
model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # Assuming it's a regression problem, change activation for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Change loss function based on your problem
              metrics=['accuracy'])  # Change metrics for classification problem

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)