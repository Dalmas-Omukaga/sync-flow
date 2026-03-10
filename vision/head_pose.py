import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        # Reference points (not used directly in simplified version)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def update(self, face_landmarks, w, h):
        # Nose x-coordinate as proxy for horizontal deviation
        nose_x = face_landmarks[1].x
        head_deviation = abs(nose_x - 0.5)  # 0=center, max ~0.5
        head_deviation_score = min(max(head_deviation * 200, 0), 100)  # scale 0-100

        return {"head_deviation": round(head_deviation_score, 1)}