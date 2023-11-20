import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import json
from common import mediapipe_detection

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

with open("./training_config.json", "r") as file:
    config_data = json.load(file)

cap = cv2.VideoCapture(config_data["video_source"])

if not cap.isOpened():
    st.write("Cannot open camera")
    exit()

col1, col2 = st.columns(2)

with col1:
    latest_frame = st.empty()
    error_text = st.empty()
    stop_button = st.button("STOP")

with col2:
    latest_result = st.empty()

# Set mediapipe model
with mp_holistic.Holistic(
    min_detection_confidence=config_data["min_detection_confidence"],
    min_tracking_confidence=config_data["min_tracking_confidence"],
) as holistic:
    # while the camera is open
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        if not ret:
            error_text.markdown(
                "<p style='color: red;'>Can't receive frame (stream end?). Exiting ...</p>",
                unsafe_allow_html=True,
            )
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Flip the image horizontally
        image = cv2.flip(image, 1)

        latest_frame.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
        latest_result.write(results)
        if stop_button:
            break

        # Draw landmarks
        # draw_styled_landmarks(image, results)

    cap.release()
    cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
