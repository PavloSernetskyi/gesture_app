import cv2
import mediapipe as mp
import numpy as np
import os

# Setup
DATA_DIR = "../data"
NUM_SAMPLES = 100  # samples per gesture

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Collecting samples
gesture_id = input("Enter numeric label for this gesture (e.g., 0 for fist, 1 for peace): ")
collected = 0
data = []

while collected < NUM_SAMPLES:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten landmark coordinates
            landmark = []
            for lm in hand_landmarks.landmark:
                landmark.extend([lm.x, lm.y, lm.z])
            data.append(landmark)
            collected += 1
            print(f"Collected: {collected}/{NUM_SAMPLES}")

    cv2.imshow("Collecting Gesture", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save data
data = np.array(data)
np.save(os.path.join(DATA_DIR, f"{gesture_id}.npy"), data)

cap.release()
cv2.destroyAllWindows()
print(f"Saved gesture data to {DATA_DIR}/{gesture_id}.npy")
