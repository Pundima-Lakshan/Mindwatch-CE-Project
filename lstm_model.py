import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import json
import numpy as np

# Read the JSON file
with open("./training_config.json", "r") as file:
    config_data = json.load(file)

# Load the datasets
with open("X_train.pkl", "rb") as file:
    X_train = pickle.load(file)

with open("X_test.pkl", "rb") as file:
    X_test = pickle.load(file)

with open("y_train.pkl", "rb") as file:
    y_train = pickle.load(file)

with open("y_test.pkl", "rb") as file:
    y_test = pickle.load(file)

# Tensorflow logging
log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

actions = np.array([item for item in config_data["actions"].split(",")])

# Create the model
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

model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

model.save("action.h5")
