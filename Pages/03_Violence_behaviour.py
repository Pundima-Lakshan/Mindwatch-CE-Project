import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide", page_title="MindWatch")

# Function to load data from CSV file
def get_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Directory Path
directory_path = r"Results\Aggressive_behavior_detection"

# List all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

# Streamlit app
st.title('Select CSV File and Display Data')

# File selection dropdown
selected_file = st.selectbox("Select a CSV file", csv_files)

# Full file path
file_path = os.path.join(directory_path, selected_file)

# Load the data
df = get_data_from_csv(file_path)

# Draw line chart for 'Violence Probability'
st.write('Violence Probability Graph')
st.line_chart(df['Violence Probability'].rename('Probability').reset_index(drop=True))

# Checkbox for showing/hiding the data table
show_data_table = st.checkbox("Show Data Table")

# Display the data table if the checkbox is ticked
if show_data_table:
    st.write("Data Table:")
    st.write(df)
