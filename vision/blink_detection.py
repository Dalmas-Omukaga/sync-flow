import time
import math

class BlinkDetector:
    def __init__(self, detector_instance=None):
        self.detector = detector_instance
        self.blink_count = 0
        self.blink_start = False
        self.start_time = time.time()
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]

    def get_distance(self, p1, p2):
        """Standard Euclidean distance for raw landmarks or pixels"""
        # If landmarks are MediaPipe objects (have x, y attributes)
        if hasattr(p1, 'x'):
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        # If landmarks are cvzone lists [x, y, z]
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def update(self, face_landmarks):
        # Extract eye points
        left = [face_landmarks[p] for p in self.left_eye_indices]

        # Vertical distances
        v1 = self.get_distance(left[1], left[5])
        v2 = self.get_distance(left[2], left[4])

        # Horizontal distance
        h = self.get_distance(left[0], left[3])

        # Eye Aspect Ratio (EAR)
        ear = (v1 + v2) / (2 * h)

        # Blink Logic
        if ear < 0.22:
            self.blink_start = True
        else:
            if self.blink_start:
                self.blink_count += 1
                self.blink_start = False

        elapsed = (time.time() - self.start_time) / 60
        blink_rate = self.blink_count / elapsed if elapsed > 0 else 0

        return {
            "blink_rate": round(blink_rate, 2),
            "ear": round(ear, 3)
        }