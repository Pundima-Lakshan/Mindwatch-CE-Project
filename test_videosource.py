import streamlit as st
import cv2
import json

# Read the JSON file
with open("./training_config.json", "r") as file:
    config_data = json.load(file)

cap = cv2.VideoCapture(config_data["video_source"])

latest_frame = st.empty()
error_text = st.empty()
stop_button = st.button("STOP")

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

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    latest_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    if stop_button:
        break

cap.release()
cv2.destroyAllWindows()
