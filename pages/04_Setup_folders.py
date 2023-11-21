import os
import numpy as np
import streamlit as st
import json
from common import CONFIG_FILE
import time

st.set_page_config(layout="wide", page_title="MindWatch")

with open(CONFIG_FILE, "r") as file:
    config_data = json.load(file)


DATA_PATH = st.text_input(
    label="Data Path", value=config_data["data_path"], help="Path to save the data"
)

actions = st.text_input(
    value=config_data["actions"], label="Actions", help="Actions to record"
)

no_sequences = st.number_input(
    value=config_data["no_sequences"],
    label="No. of Sequences",
    help="No. of Sequences",
)

sequence_length = st.number_input(
    value=config_data["sequence_length"],
    label="Sequence Length",
    help="Sequence Length in No. of Frames",
)

st.divider()

start_button = st.button("START")
state = st.empty()

DATA_PATH = os.path.join(DATA_PATH)
actions = np.array(actions.split(", "))

isError = False

if start_button:
    with st.spinner("Creating Folders..."):
        for action in actions:
            if isError:
                break

            for sequence in range(no_sequences):
                try:
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                except FileExistsError:
                    state.error("Error: Directory already exists.")
                    isError = True
                    break
                except OSError as e:
                    state.error(f"Error: {e.strerror}")
                    isError = True
                    break
                except Exception as e:
                    state.error(f"An unexpected error occurred: {str(e)}")
                    isError = True
                    break

        if not isError:
            state.success("Done")
