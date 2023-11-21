import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
import json
import numpy as np
import streamlit as st
import time
from common import CONFIG_FILE

st.set_page_config(layout="wide", page_title="MindWatch")

# Read the JSON file
with open(CONFIG_FILE, "r") as file:
    config_data = json.load(file)

DATA_PATH = os.path.join(config_data["data_path"])
actions = np.array([item for item in config_data["actions"].split(", ")])
sequence_length = config_data["sequence_length"]

start_button = st.button("START")

if start_button:
    with st.spinner("Creating labels..."):
        label_map = {label: num for num, label in enumerate(actions)}

        sequences, labels = [], []
        for action in actions:
            for sequence in np.array(
                os.listdir(os.path.join(DATA_PATH, action))
            ).astype(int):
                window = []
                for frame_num in range(sequence_length):
                    res = np.load(
                        os.path.join(
                            DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)
                        )
                    )
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])

        X = np.array(sequences)

        y = to_categorical(labels).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

        # Save the datasets

        data_sets_path = os.path.join(config_data["data_sets_path"])

        # Create the directory if it doesn't exist
        os.makedirs(data_sets_path, exist_ok=True)

        with open(os.path.join(data_sets_path, "X_train.pkl"), "wb") as file:
            pickle.dump(X_train, file)

        with open(os.path.join(data_sets_path, "X_test.pkl"), "wb") as file:
            pickle.dump(X_test, file)

        with open(os.path.join(data_sets_path, "y_train.pkl"), "wb") as file:
            pickle.dump(y_train, file)

        with open(os.path.join(data_sets_path, "y_test.pkl"), "wb") as file:
            pickle.dump(y_test, file)

        time.sleep(1)

    st.success("Data saved")
