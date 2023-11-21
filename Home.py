import streamlit as st
import json
import pandas as pd
import os

# Initialize error variable
error = False

# Title
st.title("Home")

# General Settings
st.subheader('General Settings', divider='grey')

# Input file directory
input_directory = st.text_input("Input File(s) Directory:")
if input_directory and not os.path.isdir(input_directory):
    st.error('The provided folder does not exist. Please provide a valid folder path.')
    error = True

# Output file directory
output_directory = st.text_input("Output File(s) Directory:", help='hello')
if output_directory and not os.path.isdir(output_directory):
    st.error('The provided folder does not exist. Please provide a valid folder path.')
    error = True

# Checkbox selection
st.write("Select Required Detection(s):")

Shaking_Detection_cb = st.checkbox("Shaking Detection", value=True, key="Shaking_Detection_disabled")
Gaze_Detection_cb = st.checkbox("Gaze Detection", value=True, key="Gaze_Detection_disabled")
Aggressive_Behavior_Detection_cb = st.checkbox("Aggressive Behavior Detection", value=True, key="Aggressive_Behavior_Detection_disabled")
Laying_Detection_cb = st.checkbox("Laying Detection", value=True, key="Laying_Detection_disabled")

if not(Shaking_Detection_cb or Gaze_Detection_cb or Aggressive_Behavior_Detection_cb or Laying_Detection_cb):
    st.error('Select at least one detection.')
    error = True

# Shaking Detection Settings
st.subheader('Shaking Detection Settings', divider='grey')


# Gaze Detection Settings
st.subheader('Gaze Detection Settings', divider='grey')

# Aggressive Behavior Detection Settings
st.subheader('Aggressive Behavior Detection Settings', divider='grey')

Aggressive_Behavior_Detection_model = st.radio(
    "Model:",
    ["16", "32"],
    captions = ["More accurate, less speed", "Less accurate, more speed"],
    disabled=not(Aggressive_Behavior_Detection_cb))

Aggressive_Behavior_Detection_frames_to_analyze = st.slider(
    'Frames to analyze:', 
    1, 100, 10,
    disabled=not(Aggressive_Behavior_Detection_cb))


#Laying Detection Settings
st.subheader('Laying Detection Settings', divider='grey')

# Button to generate JSON file
if st.button("Generate JSON", disabled=error):
    if not(input_directory and output_directory):
        st.error('Complete both folder paths')
    else:
        # Create a dictionary based on the selections
        data = {
            "Input Directory": input_directory,
            "Output Directory": output_directory,
            "Shaking Detection": Shaking_Detection_cb,
            "Gaze Detection": Gaze_Detection_cb,
            "Aggressive Behavior Detection": Aggressive_Behavior_Detection_cb,
            "Laying Detection": Laying_Detection_cb
        }

        # Generate JSON file
        output_json_file = "output.json"
        with open(output_json_file, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success(f"JSON file '{output_json_file}' generated successfully.")

# Display the JSON file if it exists
if "output_json_file" in locals():
    st.write("Generated JSON:")
    with open(output_json_file, "r") as json_file:
        st.json(json.load(json_file))