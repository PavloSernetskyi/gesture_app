import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib

# Load trained model
model = joblib.load("models/gesture_model.pkl")

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

st.title("🖐️ Real-Time Gesture Recognition")
st.markdown("Show your hand to the camera and perform a trained gesture.")

# Start webcam
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    success, frame = cap.read()
    if not success:
        st.write("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features
            landmark = []
            for lm in hand_landmarks.landmark:
                landmark.extend([lm.x, lm.y, lm.z])
            if len(landmark) == 63:
                data = np.array(landmark).reshape(1, -1)
                prediction = model.predict(data)[0]
                confidence = max(model.predict_proba(data)[0])
                cv2.putText(frame, f'Gesture: {prediction} ({confidence:.2f})',
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
