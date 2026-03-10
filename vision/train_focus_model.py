import os
import glob
import time
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


RAW_DATA_PATH = "data/raw/*.csv"
PROCESSED_DATA_PATH = "data/processed/focus_dataset.csv"

timestamp = int(time.time())

MODEL_PATH = "model/focus_model.pkl"
VERSIONED_MODEL_PATH = f"model/focus_model_{timestamp}.pkl"

SCALER_PATH = "model/scaler.pkl"


class FocusModelTrainer:

    def __init__(self):
        self.dataset = None


    def load_raw_sessions(self):

        print("[STEP 1] Loading raw sessions...")

        files = glob.glob(RAW_DATA_PATH)

        if len(files) == 0:
            print("[ERROR] No raw data found in data/raw/")
            exit()

        dataframes = []

        for f in files:
            try:
                df = pd.read_csv(f)

                if df.empty:
                    print(f"[WARN] Skipping empty file: {f}")
                    continue

                print("Loading:", f)
                dataframes.append(df)

            except pd.errors.EmptyDataError:
                print(f"[WARN] Skipping corrupted file: {f}")

        self.dataset = pd.concat(dataframes, ignore_index=True)

        print("[INFO] Combined dataset size:", self.dataset.shape)


    def clean_dataset(self):

        print("[STEP 2] Cleaning dataset...")

        df = self.dataset.copy()

        df = df.dropna()

        df = df[
            (df["blink_rate"] >= 0) &
            (df["blink_rate"] <= 60) &
            (df["gaze_score"] >= 0) &
            (df["gaze_score"] <= 100) &
            (df["head_deviation"] >= 0) &
            (df["head_deviation"] <= 1)
        ]

        if df["focus_label"].dtype == object:

            label_map = {
                "very_distracted": 0,
                "distracted": 1,
                "neutral": 2,
                "slightly_focused": 3,
                "highly_focused": 4
            }

            df["focus_label"] = df["focus_label"].map(label_map)

            df = df.dropna(subset=["focus_label"])

            df["focus_label"] = df["focus_label"].astype(int)

        self.dataset = df

        print("[INFO] Clean dataset size:", self.dataset.shape)


    def save_processed_dataset(self):

        print("[STEP 3] Saving processed dataset...")

        os.makedirs("data/processed", exist_ok=True)

        self.dataset.to_csv(PROCESSED_DATA_PATH, index=False)

        print("[SAVED]", PROCESSED_DATA_PATH)


    def train_model(self):

        print("[STEP 4] Training focus model...")

        X = self.dataset[[
            "blink_rate",
            "gaze_score",
            "head_deviation"
        ]]

        y = self.dataset["focus_label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)

        print("[RESULT] Mean Absolute Error:", round(mae, 3))

        os.makedirs("model", exist_ok=True)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(model, VERSIONED_MODEL_PATH)

        joblib.dump(scaler, SCALER_PATH)

        print("[SAVED MODEL]", MODEL_PATH)
        print("[SAVED VERSIONED MODEL]", VERSIONED_MODEL_PATH)
        print("[SAVED SCALER]", SCALER_PATH)

        print("\nDataset statistics:")
        print(self.dataset.describe())

        importance = model.feature_importances_

        print("\nFeature Importance:")
        for name, score in zip(X.columns, importance):
            print(f"{name}: {score:.3f}")


    def run(self):

        self.load_raw_sessions()
        self.clean_dataset()
        self.save_processed_dataset()
        self.train_model()


if __name__ == "__main__":

    trainer = FocusModelTrainer()
    trainer.run()