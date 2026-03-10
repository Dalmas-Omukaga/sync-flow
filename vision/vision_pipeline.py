import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
import cv2
import time
import subprocess
import numpy as np
import socketio  # pip install "python-socketio[client]"
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe as mp
import joblib
import csv
import pandas as pd  

# Modular Imports
from vision.blink_detection import BlinkDetector
from vision.gaze_tracking import GazeTracker
from vision.head_pose import HeadPoseEstimator
from vision.attention_model import AttentionModel

# --- ARCH LINUX / MediaPipe SAFETY INJECTOR ---
if not hasattr(mp, 'solutions'):
    import mediapipe.python.solutions as solutions
    mp.solutions = solutions

class VisionPipeline:
    def __init__(self, server_url="http://localhost:5000"):
        self.frame_count = 0
        
        self.LIVE_PATH = "data/live/focus_predictions_live.csv"
        os.makedirs("data/live", exist_ok=True)
        # 1. Hardware & Core AI
        self.cap = cv2.VideoCapture(0)

        # Initialize FaceMeshDetector
        self.detector = FaceMeshDetector(maxFaces=1)
        self.detector.faceMesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 2. Modular Feature Extractors
        self.blink_mod = BlinkDetector(self.detector)
        self.gaze_mod = GazeTracker()
        self.pose_mod = HeadPoseEstimator()

        # 3. Intelligence Layer
        self.attention_model = AttentionModel()  # fallback
        self.focus_model = None
        self.scaler = None
        self.load_focus_model()

        # 4. Networking (SocketIO Client)
        self.sio = socketio.Client()
        self.server_url = server_url
        self.connected = False
        self.connect_to_server()

        # 5. FPS tracking
        self.prev_time = time.time()

        self.last_send_time = time.time()
        self.send_interval = 10

        self.score_buffer = []

        # 6. CSV Logging for live dataset
        self.log_file = "data/raw/live_focus_log.csv"
        os.makedirs("data/raw", exist_ok=True)
        if not os.path.isfile(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "blink_rate",
                    "gaze_score",
                    "head_deviation",
                    "focus_score",
                    "focus_label"
                ])

    # ----------------------- MODEL LOADING -----------------------
    def load_focus_model(self):
        try:
            self.focus_model = joblib.load("model/focus_model.pkl")
            self.scaler = joblib.load("model/scaler.pkl")
            print("[MODEL] Loaded focus model from model/focus_model.pkl")
        except Exception as e:
            print(f"[WARN] Could not load focus model, using fallback AttentionModel: {e}")
            self.focus_model = None
            self.scaler = None


    def sync_to_cloud(self):
        try:
            # 1. Add the log file
            subprocess.run(["git", "add", "data/live/live_focus_log.csv"], check=True)
            # 2. Commit with a timestamp (to avoid merge conflicts)
            subprocess.run(["git", "commit", "-m", f"data-sync: {time.time()}"], check=True)
            # 3. Push to GitHub
            subprocess.run(["git", "push", "origin", "main"], check=True)
            print("🚀 Sync Successful: Data beamed to Cloud!")
        except Exception as e:
            print(f"❌ Sync Failed: {e}")

    # ----------------------- SERVER -----------------------
    def connect_to_server(self):
        try:
            self.sio.connect(self.server_url)
            self.connected = True
            print(f"[CONNECTED] Streaming to {self.server_url}")
        except Exception:
            print(f"[WARN] Backend server not found. Running in standalone mode.")

    # ----------------------- FOCUS LABEL MAPPER -----------------------
    def focus_label_from_score(self, score):
        if score >= 80:
            return "highly_focused"
        elif score >= 60:
            return "slightly_Focused"
        elif score >= 40:
            return "Neutral"
        elif score >= 20:
            return "distracted"
        else:
            return "very_dist_distracted"

    # ----------------------- MAIN LOOP -----------------------
    def run(self):
        print("[INFO] Sync-Flow Vision Pipeline Active. ESC to quit.")
        last_model_reload = time.time()

        while True:
            self.frame_count += 1
            success, img = self.cap.read()
            if not success: break

            h, w, _ = img.shape
            img, faces = self.detector.findFaceMesh(img, draw=False)

            if faces and self.detector.results and self.detector.results.multi_face_landmarks:
                raw_landmarks = self.detector.results.multi_face_landmarks[0].landmark
                
                # --- Step A: Feature Extraction ---
                blink_data = self.blink_mod.update(faces[0])
                gaze_data = self.gaze_mod.update(raw_landmarks, w, h)
                pose_data = self.pose_mod.update(raw_landmarks, w, h)

                # --- Step B: Data Alignment ---
                live_gaze = gaze_data.get("gaze_score", 0)
                if 0 < live_gaze <= 1.0: live_gaze *= 100 

                raw_head = pose_data.get("head_deviation", 0)
                live_head = min(1.0, raw_head / 500.0) 

                combined_features = {
                    "blink_rate": blink_data.get("blink_rate", 0),
                    "gaze_score": live_gaze,
                    "head_deviation": live_head
                }

                # --- Step C: Focus Prediction ---
                try:
                    if self.focus_model and self.scaler:
                        X = pd.DataFrame([[
                            combined_features["blink_rate"],
                            combined_features["gaze_score"],
                            combined_features["head_deviation"]
                        ]], columns=["blink_rate", "gaze_score", "head_deviation"])
                        
                        X_scaled = self.scaler.transform(X)
                        raw_score = float(self.focus_model.predict(X_scaled)[0]) * 25
                    else:
                        raw_score = self.attention_model.compute_focus(combined_features)
                    
                    self.score_buffer.append(raw_score)
                    if len(self.score_buffer) > 15: self.score_buffer.pop(0)
                    focus_score = sum(self.score_buffer) / len(self.score_buffer)
                except Exception as e:
                    focus_score = self.attention_model.compute_focus(combined_features)

                focus_label = self.focus_label_from_score(focus_score)

                # --- Step D: Throttled Logging & Sync ---
                current_time = time.time()
                
                # Trigger Cloud Sync (Fixed Variables)
                if self.frame_count % 900 == 0:
                    self.sync_to_cloud()

                # Periodic Model Reload (Fixed to avoid 30x reload)
                if current_time - last_model_reload > 300:
                    self.load_focus_model()
                    last_model_reload = current_time

                if current_time - self.last_send_time > self.send_interval:
                    log_data = {
                        "timestamp": current_time,
                        "blink_rate": combined_features["blink_rate"],
                        "gaze_score": combined_features["gaze_score"],
                        "head_deviation": combined_features["head_deviation"],
                        "focus_score": round(focus_score, 2),
                        "focus_label": focus_label
                    }
                    
                    # Update Live CSV (For Streamlit Cloud)
                    df = pd.DataFrame([log_data])
                    df.to_csv(self.LIVE_PATH, mode='a', header=not os.path.exists(self.LIVE_PATH), index=False)

                    if self.connected:
                        self.sio.emit('vision_data', log_data)
                    
                    self.last_send_time = current_time

                # --- Step E: Render UI ---
                # Calculate bounding box for the reticle
                face_points = np.array(faces[0])
                min_c, max_c = np.min(face_points, axis=0), np.max(face_points, axis=0)
                computed_bbox = [int(min_c[0]), int(min_c[1]), int(max_c[0]-min_c[0]), int(max_c[1]-min_c[1])]
                
                self.draw_reticle(img, computed_bbox, focus_score)
                self.render_overlay(img, focus_score, focus_label, combined_features)

            cv2.imshow("Sync-Flow Integrated Vision", img)
            if cv2.waitKey(1) & 0xFF == 27: break

        self.cleanup()

    # ----------------------- RENDER -----------------------
    def draw_reticle(self, img, bbox, score):
        """Draws corner brackets that track the face movement"""
        x, y, w, h = bbox
        # Match color to focus state
        color = (127, 255, 0) if score > 75 else (0, 215, 255) if score > 45 else (50, 50, 255)
        d, t = 25, 2 # Length and thickness
        # Top Left
        cv2.line(img, (x, y), (x + d, y), color, t)
        cv2.line(img, (x, y), (x, y + d), color, t)
        # Top Right
        cv2.line(img, (x + w, y), (x + w - d, y), color, t)
        cv2.line(img, (x + w, y), (x + w, y + d), color, t)
        # Bottom Left
        cv2.line(img, (x, y + h), (x + d, y + h), color, t)
        cv2.line(img, (x, y + h), (x, y + h - d), color, t)
        # Bottom Right
        cv2.line(img, (x + w, y + h), (x + w - d, y + h), color, t)
        cv2.line(img, (x + w, y + h), (x + w, y + h - d), color, t)

    def render_overlay(self, img, score, label, data):
        h, w, _ = img.shape
        # 1. Semi-transparent HUD Header
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # 2. Dynamic Color Selection
        color = (127, 255, 0) if score > 75 else (0, 215, 255) if score > 45 else (50, 50, 255)
        
        # 3. Glassmorphism Progress Bar
        bar_x, bar_y, bar_w, bar_h = 40, 70, 300, 12
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * (max(0, min(100, score)) / 100))
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        # 4. Main Focus Labels
        cv2.putText(img, "SYNC-FLOW LIVE ANALYTICS", (40, 45), 1, 0.8, (200, 200, 200), 1)
        cv2.putText(img, f"{int(score)}%", (bar_x + bar_w + 15, bar_y + 12), 1, 1.5, color, 2)
        cv2.putText(img, label.upper().replace("_", " "), (40, 105), 1, 0.8, color, 1)

        # 5. Right-Side Biometrics
        m_x = w - 230
        cv2.putText(img, f"GAZE: {data.get('gaze_score', 0)*100:.1f}%", (m_x, 45), 1, 0.8, (200,200,200), 1)
        cv2.putText(img, f"BLINK: {data.get('blink_rate', 0)}/m", (m_x, 75), 1, 0.8, (200,200,200), 1)
        cv2.putText(img, f"HEAD: {data.get('head_deviation', 0)*100:.1f}%", (m_x, 105), 1, 0.8, (200,200,200), 1)

    def render_fps(self, img):
        curr_time = time.time()
        # Prevent division by zero if frames are too fast
        time_diff = curr_time - self.prev_time
        if time_diff > 0:
            fps = 1 / time_diff
            self.prev_time = curr_time
            # Draw FPS on top right
            cv2.putText(img, f"FPS: {fps:.1f}", (img.shape[1] - 150, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ----------------------- CLEANUP -----------------------
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        try:
            if self.connected:
                self.sio.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    pipeline = VisionPipeline()
    pipeline.run()
