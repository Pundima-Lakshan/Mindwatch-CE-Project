# Run the logic function in logic.py in a streamlit app

import streamlit as st
import cv2

from logic import test_webcam, test_detection, setup_folders, test_detection_with_landmark_drawn

# streamlit basic setup with one sidebar and main area
st.sidebar.title("Action Recognition App")
st.sidebar.markdown("App to recognize actions from webcam feed and train LSTM model using mediapipe and tensorflow")

in_browser_feed =  st.sidebar.checkbox("In browser video feed", True)

# buttons to run different functions from logic.py
if st.sidebar.button("Test Webcam"):
    test_webcam(in_browser_feed)

if st.sidebar.button("Test Detection"):
    test_detection(in_browser_feed)

if st.sidebar.button("Test Detection with Landmark Drawn"):
    test_detection_with_landmark_drawn(in_browser_feed)

if st.sidebar.button("Setup Folders"):
    setup_folders(in_browser_feed)

if st.sidebar.button("Clear all CV2 Windows"):
    cv2.destroyAllWindows()