# train_sensor_model.py
# This file trains a simple ML model using sensor data (rainfall, vibration, temp)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("sensor_data.csv")

X = df[['rainfall', 'vibration', 'temp']]
y = df['label']

# Train ML model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Save model
with open("models/sensor_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Sensor model saved as models/sensor_model.pkl")
