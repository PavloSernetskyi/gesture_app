import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model
model = joblib.load("../models/gesture_model.pkl")

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks as a flat array
            landmark = []
            for lm in hand_landmarks.landmark:
                landmark.extend([lm.x, lm.y, lm.z])

            # Predict gesture
            data = np.array(landmark).reshape(1, -1)
            prediction = model.predict(data)[0]
            confidence = max(model.predict_proba(data)[0])

            # Show prediction
            cv2.putText(img, f'Gesture: {prediction} ({confidence:.2f})',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Live Prediction", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
