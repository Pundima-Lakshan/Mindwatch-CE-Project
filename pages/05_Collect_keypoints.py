import os
import cv2
import mediapipe as mp
import numpy as np
import json
import streamlit as st
from common import (
    CONFIG_FILE,
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
)

st.set_page_config(layout="wide", page_title="MindWatch")

with open(CONFIG_FILE, "r") as file:
    config_data = json.load(file)

# Path for exported data, numpy arrays
DATA_PATH = os.path.join(config_data["data_path"])

actions = np.array([item for item in config_data["actions"].split(", ")])
no_sequences = config_data["no_sequences"]
sequence_length = config_data["sequence_length"]
start_folder = config_data["start_folder"]

cap = cv2.VideoCapture(config_data["video_source"])

start_button = st.button("START")

isStopped = False

mp_holistic = mp.solutions.holistic  # Holistic model

if start_button:
    state_text = st.empty()
    latest_frame = st.empty()
    error_text = st.empty()
    stop_button = st.button("STOP")

    with mp_holistic.Holistic(
        min_detection_confidence=config_data["min_detection_confidence"],
        min_tracking_confidence=config_data["min_tracking_confidence"],
    ) as holistic:
        if stop_button:
            isStopped = True

        for action in actions:
            if isStopped:
                break

            for sequence in range(start_folder, start_folder + no_sequences):
                if isStopped:
                    break

                for frame_num in range(sequence_length):
                    if isStopped:
                        break

                    if cap.isOpened():
                        ret, frame = cap.read()
                    else:
                        error_text.error(
                            "Can't receive frame (stream end?). Exiting ... closed"
                        )
                        error_text.error(f"{action}, {sequence}, {frame_num}")
                        break

                    if not ret:
                        error_text.error(
                            "Can't receive frame (stream end?). Exiting ... frame"
                        )
                        error_text.error(f"{action}, {sequence}, {frame_num}")
                        break

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    # wait logic
                    if frame_num == 0:
                        # Flip the image horizontally
                        image = cv2.flip(image, 1)

                        # Calculate the width and height of the text box
                        (text_width, text_height) = cv2.getTextSize(
                            "STARTING COLLECTION", cv2.FONT_HERSHEY_SIMPLEX, 1, 4
                        )[0]

                        # Set the text start position
                        text_x, text_y = 120, 200

                        # Draw a rectangle filled with color
                        cv2.rectangle(
                            image,
                            (text_x - 15, text_y + 15),
                            (text_x + text_width + 15, text_y - text_height - 15),
                            (102, 0, 255),
                            -1,
                        )

                        cv2.putText(
                            img=image,
                            text="STARTING COLLECTION",
                            org=(120, 200),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(255, 255, 255),
                            thickness=4,
                            lineType=cv2.LINE_AA,
                        )

                        latest_frame.image(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB"
                        )
                        cv2.waitKey(config_data["wait_time"])
                    else:
                        # Flip the image horizontally
                        image = cv2.flip(image, 1)

                        latest_frame.image(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB"
                        )

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(
                        DATA_PATH, action, str(sequence), str(frame_num)
                    )

                    np.save(npy_path, keypoints)

                    state_text.info(
                        "Collecting frames for {} Video Number {}".format(
                            action, sequence
                        )
                    )

        cap.release()
        st.info("Data Collection Completed")

    cap.release()
