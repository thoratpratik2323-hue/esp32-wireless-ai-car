"""
Custom AI Model Trainer for Smart Car
-------------------------------------
As an AI/ML student, you can use this script to experiment with different 
Machine Learning models. 

This script loads the data you collected ('training_data.csv'), trains a custom 
model, and saves it as 'car_ai_model.pkl'. The main dashboard will automatically 
load your custom trained model to drive the physical car!

Make sure you have collected training data before running this script.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import different classifiers to experiment with
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 1. Load the logged dataset
DATA_FILE = "training_data.csv"

if not os.path.exists(DATA_FILE):
    print(f"Error: Dataset '{DATA_FILE}' not found. Run the dashboard and log some data first!")
    exit()

df = pd.read_csv(DATA_FILE)
print(f"Loaded dataset successfully! Total samples: {len(df)}")

# 2. Separate features (X) and labels (y)
# Features: Left, Center, Right sensor distances
X = df[["dist_left", "dist_center", "dist_right"]].values
# Target Labels: Actions ('F' = Forward, 'L' = Left, 'R' = Right, 'B' = Backward)
y = df["action"].values

# Split data into training and testing sets to check performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------
# 3. CONFIGURE THE AI MODEL (Neural Network / MLP)
# -------------------------------------------------------------

# We are using an Artificial Neural Network (Multi-Layer Perceptron)
# - Input Layer: 3 input features (Left, Center, Right distances)
# - Hidden Layer 1: 16 neurons with Rectified Linear Unit (ReLU) activation
# - Hidden Layer 2: 8 neurons with Rectified Linear Unit (ReLU) activation
# - Output Layer: Action predictions ('F', 'B', 'L', 'R')
# - Solver: 'adam' (stochastic gradient descent optimizer)
# - verbose=True: prints the loss value at each epoch to monitor learning convergence

print("\nInitializing Artificial Neural Network (MLP)...")
model = MLPClassifier(
    hidden_layer_sizes=(16, 8), 
    activation='relu', 
    solver='adam', 
    max_iter=1000, 
    verbose=True, 
    random_state=42
)

# Alternative models for reference (uncomment to test):
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# -------------------------------------------------------------
# 4. Train the selected Model
# -------------------------------------------------------------
print("\nTraining the Neural Network on your dataset...")
model.fit(X_train, y_train)

# 5. Evaluate the Model's accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Evaluation:")
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 6. Save the model as 'car_ai_model.pkl'
# The main dashboard will read this file automatically on startup
model_filename = "car_ai_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)

print(f"\nSuccess! Your custom AI model has been saved as '{model_filename}'.")
print("Now restart or run the dashboard, press 'O' (Auto Mode), and the car will use your custom model!")
