import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from scipy import stats
import cv2
import mediapipe as mp
import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
from common import (
    CONFIG_FILE,
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
    prob_viz,
)

st.set_page_config(layout="wide", page_title="MindWatch")

# Read the JSON file
with open(CONFIG_FILE, "r") as file:
    config_data = json.load(file)

actions = np.array([item for item in config_data["actions"].split(", ")])
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))

model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

weights_path = os.path.join(config_data["weights_path"])

model.load_weights(os.path.join(weights_path, "action.h5"))


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

isStopped = False

col1, col2 = st.columns([2, 1])

with col1:
    latest_frame = st.empty()

with col2:
    start_button = st.button("START", use_container_width=True)
    state_text = st.empty()
    error_text = st.empty()
    stop_button = st.button("STOP", use_container_width=True)

cap = cv2.VideoCapture(config_data["video_source"])

if not cap.isOpened():
    error_text.error("Can't open video source.")
    st.stop()

if start_button:
    # Set mediapipe model
    with mp_holistic.Holistic(
        min_detection_confidence=config_data["min_detection_confidence"],
        min_tracking_confidence=config_data["min_tracking_confidence"],
    ) as holistic:
        if stop_button:
            isStopped = True

        while cap.isOpened():
            if isStopped:
                break

            ret, frame = cap.read()

            if not ret:
                error_text.error("Can't receive frame (stream end?). Exiting ... frame")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)
                state_text.info(sentence[-1])

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                image,
                " ".join(sentence),
                (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            latest_frame.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_column_width=True,
            )

        cap.release()

    cap.release()
