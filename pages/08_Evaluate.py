import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import json
import numpy as np
from common import CONFIG_FILE
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

st.set_page_config(layout="centered", page_title="MindWatch")

# Read the JSON file
with open(CONFIG_FILE, "r") as file:
    config_data = json.load(file)

actions = np.array([item for item in config_data["actions"].split(", ")])

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

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

cm = confusion_matrix(ytrue, yhat)
acs = accuracy_score(ytrue, yhat)

# Convert to DataFrame for easier plotting
cm_df = pd.DataFrame(cm)

plt.figure(figsize=(10, 7))

# Use seaborn's heatmap function to visualize the confusion matrix
sns.heatmap(cm_df, annot=True, fmt="g", cmap="Blues")
plt.title("Confusion matrix")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

st.pyplot(plt, use_container_width=True)

# st.write(multilabel_confusion_matrix(ytrue, yhat))
st.warning(f"Actions {actions}")
st.info(f"Accuracy measure: {acs}")
