import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import json
import numpy as np
from common import CONFIG_FILE, delete_all_files_and_dirs, delete_all_weights
import streamlit as st
from tensorboard import notebook
from streamlit_tensorboard import st_tensorboard


st.set_page_config(layout="wide", page_title="MindWatch")

# Read the JSON file
with open(CONFIG_FILE, "r") as file:
    config_data = json.load(file)

data_sets_path = os.path.join(config_data["data_sets_path"])

# Load the datasets
try:
    with open(os.path.join(data_sets_path, "X_train.pkl"), "rb") as file:
        X_train = pickle.load(file)

    with open(os.path.join(data_sets_path, "X_test.pkl"), "rb") as file:
        X_test = pickle.load(file)

    with open(os.path.join(data_sets_path, "y_train.pkl"), "rb") as file:
        y_train = pickle.load(file)

    with open(os.path.join(data_sets_path, "y_test.pkl"), "rb") as file:
        y_test = pickle.load(file)
except Exception as e:
    st.error(f"Error loading data sets: {e}")
    st.stop()


# Tensorflow logging
log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

col1, col2, col3 = st.columns(3)
container = st.container()

with col1:
    start_button = st.button("START TRAINING", use_container_width=True)
with col2:
    delete_logs = st.button("DELETE LOGS", use_container_width=True)
with col3:
    delete_weights = st.button("DELETE WEIGHTS", use_container_width=True)

# Display the TensorBoard
st_tensorboard(logdir=log_dir, port=6006, height=700)

if delete_logs:
    delete_all_files_and_dirs(log_dir)
    with container:
        st.success("Logs deleted")

weights_path = os.path.join(config_data["weights_path"])

if delete_weights:
    delete_all_weights(weights_path)
    with container:
        st.success("Weights deleted")

if start_button:
    with container:
        with st.spinner("Creating labels..."):
            actions = np.array([item for item in config_data["actions"].split(", ")])

            # Create the model
            model = Sequential()
            model.add(
                LSTM(
                    64, return_sequences=True, activation="relu", input_shape=(30, 1662)
                )
            )
            model.add(LSTM(128, return_sequences=True, activation="relu"))
            model.add(LSTM(64, return_sequences=False, activation="relu"))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(actions.shape[0], activation="softmax"))

            model.compile(
                optimizer="Adam",
                loss="categorical_crossentropy",
                metrics=["categorical_accuracy"],
            )

            model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

            # Create the directory if it doesn't exist
            os.makedirs(weights_path, exist_ok=True)

            model.save(os.path.join(weights_path, "action.h5"))

        st.success("Model saved")
