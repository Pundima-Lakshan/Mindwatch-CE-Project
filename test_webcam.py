import streamlit as st
import cv2

cap = cv2.VideoCapture(0)

latest_frame = st.empty()
error_text = st.empty()
stop_button = st.button("STOP")

# while the camera is open
while cap.isOpened():
    # Read feed
    ret, frame = cap.read()

    if not ret:
        error_text.markdown("<p style='color: red;'>Can't receive frame (stream end?). Exiting ...</p>", unsafe_allow_html=True)
        print("Can't receive frame (stream end?). Exiting ...")
        break

    latest_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    if stop_button:
        break

cap.release()
cv2.destroyAllWindows()