import time
import subprocess
import os

DATA_FILE = "data/raw/live_focus_log.csv"

MIN_ROWS = 500   # minimum dataset size before retraining
CHECK_INTERVAL = 600  # seconds

def dataset_size():
    if not os.path.exists(DATA_FILE):
        return 0

    with open(DATA_FILE) as f:
        return sum(1 for _ in f) - 1


def run_retrain():
    print("[AUTO TRAIN] Processing dataset")
    subprocess.run(["python", "-m", "vision.process_dataset"])

    print("[AUTO TRAIN] Training model")
    subprocess.run(["python", "-m", "vision.train_focus_model"])


if __name__ == "__main__":

    while True:

        rows = dataset_size()
        print(f"[AUTO TRAIN] Dataset rows: {rows}")

        if rows > MIN_ROWS:
            run_retrain()

        time.sleep(CHECK_INTERVAL)