import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
import json
import numpy as np

# Read the JSON file
with open("./training_config.json", "r") as file:
    config_data = json.load(file)

DATA_PATH = os.path.join(config_data["data_path"])
actions = np.array([item for item in config_data["actions"].split(",")])
sequence_length = config_data["sequence_length"]

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
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

import pickle

# Save the datasets
with open("X_train.pkl", "wb") as file:
    pickle.dump(X_train, file)

with open("X_test.pkl", "wb") as file:
    pickle.dump(X_test, file)

with open("y_train.pkl", "wb") as file:
    pickle.dump(y_train, file)

with open("y_test.pkl", "wb") as file:
    pickle.dump(y_test, file)
