import numpy as np


class GazeTracker:

    def __init__(self):

        # MediaPipe indices
        self.LEFT_IRIS = [473, 474, 475, 476, 477]

        self.LEFT_EYE_CORNERS = [362, 263]

        self.history = []

        self.ratio_history = []


    def update(self, face_landmarks, frame_w, frame_h):

        ratio = 0.5
        gaze_direction = "center"
        focus_score = 50

        try:

            mesh_points = np.array(
                [[int(p.x * frame_w), int(p.y * frame_h)] for p in face_landmarks]
            )

            # --- iris center ---
            iris_points = mesh_points[self.LEFT_IRIS]
            iris_center = iris_points.mean(axis=0)

            # --- eye corners ---
            left_corner = mesh_points[self.LEFT_EYE_CORNERS[0]]
            right_corner = mesh_points[self.LEFT_EYE_CORNERS[1]]

            eye_x_min = min(left_corner[0], right_corner[0])
            eye_x_max = max(left_corner[0], right_corner[0])

            eye_width = eye_x_max - eye_x_min

            if eye_width > 0:

                ratio = (iris_center[0] - eye_x_min) / eye_width

            ratio = np.clip(ratio, 0.0, 1.0)

            # --- gaze direction classification ---
            if ratio < 0.35:
                gaze_direction = "left"

            elif ratio > 0.65:
                gaze_direction = "right"

            else:
                gaze_direction = "center"

            # --- 1. COMFORT ZONE LOGIC ---
            diff = abs(ratio - 0.5)
            if diff <= 0.05:
                position_score = 100
            else:
                position_score = 100 * (1 - (diff - 0.05) / 0.45)

            # --- 2. STABILITY VARIANCE (JITTER) ---
            self.ratio_history.append(ratio)
            if len(self.ratio_history) > 15:
                self.ratio_history.pop(0)
            
            raw_variance = np.var(self.ratio_history) if len(self.ratio_history) > 5 else 0
            stability_penalty = min(raw_variance * 5000, 40) 
            
            # --- 3. FINAL INTEGRATED SCORE (Keep this, delete the lines below it) ---
            focus_score = position_score - stability_penalty
            focus_score = np.clip(focus_score, 0, 100)


            # --- focus score calculation ---
            # center = 100
            # edges = 0

            #focus_score = 100 * (1 - abs(ratio - 0.5) * 2)

            #focus_score = np.clip(focus_score, 0, 100)

            # --- smoothing ---
            self.history.append(focus_score)

            if len(self.history) > 10:
                self.history.pop(0)

            focus_score = np.mean(self.history)


            # --- detect looking away ---
            looking_away = False

            if ratio < 0.1 or ratio > 0.9:
                looking_away = True
                focus_score = focus_score * 0.4


            # --- debug ---
            print(
                f"Gaze Ratio: {ratio:.2f} | "
                f"Direction: {gaze_direction} | "
                f"Focus: {focus_score:.1f}"
            )


        except Exception as e:

            print("Gaze Tracker Error:", e)

            gaze_direction = "unknown"
            focus_score = 0


        return {

            "gaze_ratio": ratio,
            "gaze_direction": gaze_direction,
            "gaze_score": round(focus_score, 1)

        }