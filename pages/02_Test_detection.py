import cv2
import mediapipe as mp
import streamlit as st
import json
from common import CONFIG_FILE, mediapipe_detection

st.set_page_config(layout="wide", page_title="MindWatch")

with open(CONFIG_FILE, "r") as file:
    config_data = json.load(file)

cap = cv2.VideoCapture(config_data["video_source"])

if not cap.isOpened():
    st.error("Can't open video source.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    latest_frame = st.empty()
    error_text = st.empty()
    stop_button = st.button("STOP")

with col2:
    latest_result = st.empty()

mp_holistic = mp.solutions.holistic  # Holistic model

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
            error_text.error("Can't receive frame (stream end?). Exiting ...")
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Flip the image horizontally
        image = cv2.flip(image, 1)

        latest_frame.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
        latest_result.write(results)

        if stop_button:
            break

    cap.release()

cap.release()
