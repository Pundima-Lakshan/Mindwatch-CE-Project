import streamlit as st
import pandas as pd
import os

# Function to load data from CSV file
def get_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Directory Path
directory_path = r"D:\Git\Mindwatch-CE-Project\Results\Sleeping"

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

# Display the data table
st.write("Data Table:")
st.write(df)

# Create line chart for 'Sleeping Probability (Yes)'
st.title('Sleeping Probability (Yes) Graph')
st.line_chart(df['sleeping Probability (Yes)'].rename('Probability (Yes)').reset_index(drop=True))

# Create line chart for 'Sleeping Probability (No)'
st.title('Sleeping Probability (No) Graph')
st.line_chart(df['sleeping Probability (No)'].rename('Probability (No)').reset_index(drop=True))
