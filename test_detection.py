import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.write("Cannot open camera")
    exit()

col1, col2 = st.columns(2)

with col1:
    stop_button = st.button("STOP")
    latest_frame = st.empty()            
    error_text = st.empty()

with col2:
    latest_result = st.empty()

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # while the camera is open
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        if not ret:
            error_text.markdown("<p style='color: red;'>Can't receive frame (stream end?). Exiting ...</p>", unsafe_allow_html=True)
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
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
