import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("dataset.csv")

X = df.drop(columns=["focus_score"])
y = df["focus_score"]

model = RandomForestRegressor()

model.fit(X, y)

joblib.dump(model, "focus_model.pkl")