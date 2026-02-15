"""Quick debug overlay — shows webcam + MediaPipe hand landmarks live."""

import cv2
import numpy as np
import mediapipe

WEBCAM_INDEX = 1  # MacBook Pro Camera

_HAND_MODEL_PATH = "hand_landmarker.task"

# Hand skeleton connections for drawing
_HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

options = mediapipe.tasks.vision.HandLandmarkerOptions(
    base_options=mediapipe.tasks.BaseOptions(model_asset_path=_HAND_MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)
landmarker = mediapipe.tasks.vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Cannot open webcam index {WEBCAM_INDEX}")
    exit(1)

print(f"Webcam {WEBCAM_INDEX} opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    h, w = frame.shape[:2]

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        # Draw landmarks
        pts = []
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            pts.append((px, py))
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

        # Draw skeleton
        for a, b in _HAND_SKELETON:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

        cv2.putText(frame, f"HAND DETECTED ({len(landmarks)} landmarks)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO HAND DETECTED",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Hand Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
