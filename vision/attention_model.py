import joblib
import numpy as np
import os

class AttentionModel:

    def __init__(self):

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.model_path = os.path.join(BASE_DIR, "model", "focus_model.pkl")
        self.scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

        self.feature_order = [
            "blink_rate",
            "gaze_score",
            "head_deviation"
        ]

        self.model = None
        self.scaler = None
        self.ml_ready = False

        self.load_artifacts()

    def load_artifacts(self):

        try:

            if os.path.exists(self.model_path):

                self.model = joblib.load(self.model_path)

                if os.path.exists(self.scaler_path):
                    self.scaler = joblib.load(self.scaler_path)

                self.ml_ready = True

                print(f"[SUCCESS] Loaded Focus Model")

            else:

                print("[ERROR] focus_model.pkl not found")

        except Exception as e:

            print(f"[ERROR] Failed to load model: {e}")

    def compute_focus(self, features):

        if not self.ml_ready:
            return 50.0

        try:

            feature_vector = np.array([[
                features.get(feature, 0)
                for feature in self.feature_order
            ]])

            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)

            prediction = self.model.predict(feature_vector)

            focus_score = float(prediction[0])

            return round(max(0, min(100, focus_score)), 2)

        except Exception as e:

            print(f"[ERROR] Inference failed: {e}")

            return 0.0