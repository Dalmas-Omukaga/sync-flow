import cv2
import csv
import os
import mediapipe as mp
import time

# Ensure your imports match your directory structure
from blink_detection import BlinkDetector
from gaze_tracking import GazeTracker
from head_pose import HeadPoseEstimator

# Using a more descriptive path consistent with your project
DATASET_PATH = "data/raw/focus_dataset.csv"

class FocusDataRecorder:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        # 1. Initialize raw MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,  # Crucial for Gaze (Iris) tracking
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 2. Vision modules - Passing 'self' or None 
        # Since these modules now expect a detector, we pass the raw face_mesh object
        # or update the modules to handle raw landmarks.
        self.blink_detector = BlinkDetector(self.face_mesh) 
        self.gaze_tracker = GazeTracker() 
        self.head_pose = HeadPoseEstimator()

        # 3. Ensure data directory and dataset file exist
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        if not os.path.exists(DATASET_PATH):
            with open(DATASET_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["blink_rate", "gaze_score", "head_deviation", "focus_label"])

    def record_sample(self, blink, gaze, head):
        print("\n--- LABELING MOMENT ---")
        print("4 → Highly Focused")
        print("3 → Slightly Focused")
        print("2 → Neutral")
        print("1 → Distracted")
        print("0 → Very Distracted")
        
        try:
            label = input("Enter label (0-4): ").strip()
            # Validate that the label is within the 0-4 range
            if label not in ['0', '1', '2', '3', '4']:
                print("Invalid label! Please enter a number between 0 and 4.")
                return

            with open(DATASET_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([blink, gaze, head, label])
            print(f"✔ Sample saved with label {label}")
        except EOFError:
            pass

    def run(self):
        print("\n--- Sync-Flow Dataset Recorder ---")
        print("Live metrics showing below. Press 's' to capture current state.")
        print("Press 'q' to quit safely.\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            blink_rate = 0
            gaze_score = 0
            head_deviation = 0

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Update features using raw MediaPipe landmarks
                # Ensure your modules' update() functions accept these
                blink_data = self.blink_detector.update(landmarks)
                gaze_data = self.gaze_tracker.update(landmarks, w, h)
                head_data = self.head_pose.update(landmarks, w, h)

                blink_rate = blink_data.get("blink_rate", 0)
                gaze_score = gaze_data.get("gaze_score", 0) # Matching your gaze class key
                head_deviation = head_data.get("head_deviation", 0)

                # UI Feedback
                color = (0, 255, 0)
                cv2.putText(frame, f"Blink Rate: {blink_rate}", (30, 40), 1, 1.5, color, 2)
                cv2.putText(frame, f"Gaze Stab: {gaze_score}", (30, 80), 1, 1.5, color, 2)
                cv2.putText(frame, f"Head Dev: {head_deviation}", (30, 120), 1, 1.5, color, 2)
                cv2.putText(frame, "Press 'S' to Save", (w-200, h-30), 1, 1, (0,0,255), 1)

            cv2.imshow("Focus Recorder", frame)
            key = cv2.waitKey(1)

            if key == ord("s"):
                # Use current values
                self.record_sample(blink_rate, gaze_score, head_deviation)

            if key == ord("q") or key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recorder = FocusDataRecorder()
    recorder.run()