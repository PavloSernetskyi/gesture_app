import joblib

model = joblib.load("../models/gesture_model.pkl")
print("Loaded model:")
print(model)

print("\nModel parameters:")
print(model.get_params())
