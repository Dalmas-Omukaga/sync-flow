import cv2
import csv
import time
import argparse
import os
import mediapipe as mp
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Modular Imports
from vision.blink_detection import BlinkDetector
from vision.gaze_tracking import GazeTracker
from vision.head_pose import HeadPoseEstimator

# -------- MediaPipe Safety Injector --------
if not hasattr(mp, "solutions"):
    import mediapipe.python.solutions as solutions
    mp.solutions = solutions

class FocusDataRecorder:
    def __init__(self, label):
        self.label = label
        self.cap = cv2.VideoCapture(0)
        
        # 1. Initialize Detector
        self.detector = FaceMeshDetector(maxFaces=1)
        
        # --- CRITICAL FIX: Unlock Iris Landmarks (468-477) ---
        self.detector.faceMesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # This fixes the IndexError: 474
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 2. Initialize Feature Extractors
        self.blink = BlinkDetector(self.detector)
        self.gaze = GazeTracker()
        self.pose = HeadPoseEstimator()

        # 3. Setup File System
        os.makedirs("data/raw", exist_ok=True)
        timestamp = int(time.time())
        self.filename = f"data/raw/focus_session_{timestamp}.csv"

        self.file = open(self.filename, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["blink_rate", "gaze_score", "head_deviation", "focus_label"])

        print(f"[RECORDER] Saving dataset to {self.filename}")
        print(f"[LABEL] Recording Focus level: {self.label}")

    def run(self):
        print("[INFO] Recording started. Press ESC to stop.")
        while True:
            success, img = self.cap.read()
            if not success:
                break

            h, w, _ = img.shape
            img, faces = self.detector.findFaceMesh(img, draw=False)

            # Only record if a face is actually detected
            if faces and self.detector.results.multi_face_landmarks:
                raw_landmarks = self.detector.results.multi_face_landmarks[0].landmark

                # Extract features
                blink_data = self.blink.update(faces[0])
                gaze_data = self.gaze.update(raw_landmarks, w, h)
                pose_data = self.pose.update(raw_landmarks, w, h)

                # Prep data row
                row = [
                    blink_data["blink_rate"],
                    gaze_data["gaze_score"],
                    pose_data["head_deviation"],
                    self.label
                ]

                # Save to CSV
                self.writer.writerow(row)

                # --- Integrated UI ---
                self.draw_ui(img, row)

            cv2.imshow("Focus Data Recorder", img)
            if cv2.waitKey(1) & 0xFF == 27: # ESC key
                break

        self.cleanup()

    def draw_ui(self, img, row):
        cv2.putText(img, f"REC LABEL: {self.label}", (20, 40), 1, 2, (0, 255, 0), 2)
        cv2.putText(img, f"Blink: {row[0]} | Gaze: {row[1]:.2f} | Pose: {row[2]:.2f}", 
                    (20, 80), 1, 1, (255, 255, 255), 1)
        cv2.putText(img, "Press ESC to Save & Exit", (20, 120), 1, 1, (0, 0, 255), 1)

    def cleanup(self):
        self.cap.release()
        self.file.close()
        cv2.destroyAllWindows()
        print(f"[DONE] Data saved to {self.filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, required=True, help="Focus label (0-4)")
    args = parser.parse_args()

    recorder = FocusDataRecorder(args.label)
    recorder.run()