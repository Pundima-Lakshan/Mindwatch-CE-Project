import streamlit as st
import json
import pandas as pd
import os
import subprocess
import datetime

st.set_page_config(layout="centered", page_title="MindWatch")

# Initialize error variable
error = False

thread_manager_path = "thread_manager.py"
output_json_file = "output.json"


# General Settings
st.subheader("General Settings", divider="grey")

# Input file directory
input_directory = st.text_input(
    "Input File(s) Directory:", help="Enter the input directory", value="D:\\Projects\\1 CEProject\\git\\Mindwatch-CE-Project\\Videos"
)
if input_directory and not os.path.isdir(input_directory):
    st.error("The provided folder does not exist. Please provide a valid folder path.")
    error = True

# Checkbox selection
st.write("Select Required Detection(s):")

Aggressive_Behavior_Detection_cb = st.checkbox(
    "Aggressive Behavior Detection",
    value=True,
    key="Aggressive_Behavior_Detection_disabled",
)

Head_Pose_Detection_cb = st.checkbox(
    "Head Pose Detection", value=True, key="Head_Pose_Detection_disabled"
)

Laying_Detection_cb = st.checkbox(
    "Laying Detection", value=True, key="Laying_Detection_disabled"
)

Mood_Detection_cb = st.checkbox(
    "Mood Detection", value=True, key="Mood_Detection_disabled"
)

Standing_On_Bed_Detection_cb = st.checkbox(
    "Standing on the Bed Detection",
    value=True,
    key="Standing_On_Bed_Detection_disabled",
)

if not (
    Mood_Detection_cb
    or Head_Pose_Detection_cb
    or Aggressive_Behavior_Detection_cb
    or Laying_Detection_cb
    or Standing_On_Bed_Detection_cb
):
    st.error("Select at least one detection.")
    error = True

# Aggressive Behavior Detection Settings
st.subheader("Aggressive Behavior Detection Settings", divider="grey")

Aggressive_Behavior_Detection_model = st.radio(
    "Model:",
    ["16", "32"],
    disabled=not (Aggressive_Behavior_Detection_cb),
    help="32-model offers more accuracy but requires additional processing time compared to 16-model",
)

Aggressive_Behavior_Detection_frames_to_analyze = st.number_input(
    "Frames to analyze:",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    disabled=not (Aggressive_Behavior_Detection_cb),
    help="number of frames to consider for calculating the final violence probability in each iteration",
)

# Head Pose Detection Settings
st.subheader("Head Pose Detection Settings", divider="grey")
Visiting_Time_Start_with_colon = st.time_input(
    "Doctor's Visiting Time Start",
    datetime.time(8, 30),
    disabled=not (Head_Pose_Detection_cb),
)
Visiting_Time_Start = str(Visiting_Time_Start_with_colon).replace(":", "")

Visiting_Time_End_with_colon = st.time_input(
    "Doctor's Visiting Time End",
    datetime.time(10, 30),
    disabled=not (Head_Pose_Detection_cb),
)
Visiting_Time_End = str(Visiting_Time_End_with_colon).replace(":", "")

# Laying Detection Settings
# st.subheader("Laying Detection Settings", divider="grey")

# Mood Detection Settings
# st.subheader("Mood Detection Settings", divider="grey")

# Button to generate JSON file
if st.button("Save Settings", disabled=error):
    if not (input_directory):
        st.error("Add the input folder path for the video(s)")
    else:
        # Create a dictionary based on the selections
        data = {
            "Input_Directory": input_directory,
            "Aggressive_Behavior_Detection": Aggressive_Behavior_Detection_cb,
            "Head_Pose_Detection": Head_Pose_Detection_cb,
            "Laying_Detection": Laying_Detection_cb,
            "Mood_Detection": Mood_Detection_cb,
            "Standing_On_Bed_Detection": Standing_On_Bed_Detection_cb,
            "Aggressive_Behavior_Detection_model": int(
                Aggressive_Behavior_Detection_model
            ),
            "Aggressive_Behavior_Detection_frames_to_analyze": Aggressive_Behavior_Detection_frames_to_analyze,
            "Visiting_Time_Start": int(Visiting_Time_Start),
            "Visiting_Time_End": int(Visiting_Time_End),
        }

        # Generate JSON file
        with open(output_json_file, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success(f"JSON file '{output_json_file}' generated successfully.")

# Display the JSON file if it exists
if st.button("Show Current Settings"):
    if "output_json_file" in locals():
        st.write("Generated JSON:")
        with open(output_json_file, "r") as json_file:
            st.json(json.load(json_file))

# Read JSON file and call temporary functions
if st.button("Start Generation", type="primary"):
    try:
        subprocess.run(["python", thread_manager_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
