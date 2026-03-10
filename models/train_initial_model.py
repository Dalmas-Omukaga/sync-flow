import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import os

# 1. Create Synthetic Training Data 
# Features: blink_rate, gaze_score, head_deviation
data = {
    'blink_rate': [15, 40, 12, 50, 10, 35],
    'gaze_score': [0.9, 0.2, 0.95, 0.1, 0.85, 0.3],
    'head_deviation': [0.02, 0.3, 0.01, 0.4, 0.05, 0.25],
    'focus_score': [95, 20, 98, 10, 90, 30] # The target labels
}

df = pd.DataFrame(data)
X = df[['blink_rate', 'gaze_score', 'head_deviation']]
y = df['focus_score']

# 2. Train the XGBoost Model
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X, y)

# 3. Save to your model directory
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/focus_model.pkl')

print("[SUCCESS] focus_model.pkl has been populated and saved!")