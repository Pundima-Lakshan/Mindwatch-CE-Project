import streamlit as st
import json
import pandas as pd
import os
import subprocess

# Initialize error variable
error = False

thread_manager_path = 'thread_manager.py'
output_json_file = "output.json"

# Title
st.title("Home")

# General Settings
st.subheader('General Settings', divider='grey')

# Input file directory
input_directory = st.text_input("Input File(s) Directory:", help = 'Enter the input directory')
if input_directory and not os.path.isdir(input_directory):
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
    disabled=not(Aggressive_Behavior_Detection_cb) ,help = '32-model offers more accuracy but requires additional processing time compared to 16-model')

Aggressive_Behavior_Detection_frames_to_analyze = st.number_input('Frames to analyze:',min_value= 1,max_value=100,value =10,step=1, help= 'number of frames to consider for calculating the final violence probability in each iteration')

#Laying Detection Settings
st.subheader('Laying Detection Settings', divider='grey')

col1, col2 = st.columns(2)


# Button to generate JSON file
if st.button("Save Settings", disabled=error):
    if not(input_directory):
        st.error('Add the input folder path for the video(s)')
    else:
        # Create a dictionary based on the selections
        data = {
            "Input Directory": input_directory,
            "Shaking Detection": Shaking_Detection_cb,
            "Gaze Detection": Gaze_Detection_cb,
            "Aggressive Behavior Detection": Aggressive_Behavior_Detection_cb,
            "Laying Detection": Laying_Detection_cb
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
if st.button("Start Generation",type="primary"):
    try:
        subprocess.run(['python', thread_manager_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        