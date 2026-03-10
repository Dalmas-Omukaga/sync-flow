import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/processed/syncflow_dataset.csv")

X = df[["blink_rate", "typing_speed", "mouse_activity"]]
y = df["focus_state"]

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X, y)

joblib.dump(model, "../models/focus_model.pkl")