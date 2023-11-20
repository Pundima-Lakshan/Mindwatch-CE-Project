import streamlit as st
import cv2

if "mw" in st.session_state:
    mw = st.session_state.mw
else:
    st.error("Please go to Home page and initialize the app.")
    st.stop()

if mw.isError:
    st.error("Something went wrong. Please restart the app.")
    st.stop()

latest_frame = st.empty()
error_text = st.empty()

if mw.cap is None:
    error_text.error("Can't open video source.")
    st.stop()

isRelease = st.button("Relase video source")

while mw.cap.isOpened():
    ret, frame = mw.cap.read()

    if not ret:
        error_text.error("Can't receive frame. Exiting...")
        st.stop()
        break

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    latest_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    if isRelease:
        break

if isRelease:
    mw.cap.release()
    cv2.destroyAllWindows()
    st.stop()
