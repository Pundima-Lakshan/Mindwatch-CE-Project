import streamlit as st
import json

# Title
st.title("JSON File Generator")

# Input file directory
input_directory = st.text_input("Input File Directory:")

# Output file directory
output_directory = st.text_input("Output File Directory:")

# Dropdown selection
st.write("Select Required Detection:")
selected_items = st.multiselect("add or remove", ["Shaking Detection", 
                                                   "Gaze Detection", 
                                                   "Aggressive Behavior Detection",
                                                   "Laying Detection"],["Shaking Detection", 
                                                   "Gaze Detection", 
                                                   "Aggressive Behavior Detection",
                                                   "Laying Detection"])

# Button to generate JSON file
if st.button("Generate JSON"):
    if input_directory and output_directory and len(selected_items) > 0:
        # Create a dictionary based on the selections
        data = {
            "Input Directory": input_directory,
            "Output Directory": output_directory,
            "Selected Items": selected_items
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
