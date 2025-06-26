import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load all gesture data
DATA_DIR = "../data"
X = []
y = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        label = int(file.split(".")[0])  # '0.npy' -> 0
        data = np.load(os.path.join(DATA_DIR, file))
        X.extend(data)
        y.extend([label] * len(data))

X = np.array(X)
y = np.array(y)

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/gesture_model.pkl")
print("Model saved to ../models/gesture_model.pkl")
