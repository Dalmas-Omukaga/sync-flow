import pandas as pd
import os

RAW_FILE = "data/raw/live_focus_log.csv"
PROCESSED_FILE = "data/processed/focus_dataset.csv"

os.makedirs("data/processed", exist_ok=True)

def process_dataset():
    if not os.path.exists(RAW_FILE):
        print("No raw dataset found.")
        return

    df = pd.read_csv(RAW_FILE)

    # Remove invalid rows
    df = df.dropna()

    # Remove impossible values
    df = df[
        (df["blink_rate"] >= 0) &
        (df["gaze_score"] >= 0) &
        (df["gaze_score"] <= 100) &
        (df["head_deviation"] >= 0) &
        (df["head_deviation"] <= 1)
    ]

    # Save processed dataset
    df.to_csv(PROCESSED_FILE, index=False)

    print(f"[DATASET] Clean dataset saved: {PROCESSED_FILE}")
    print(f"[DATASET] Rows: {len(df)}")


if __name__ == "__main__":
    process_dataset()