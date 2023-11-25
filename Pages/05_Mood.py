import streamlit as st
import pandas as pd
import ast
import os

st.set_page_config(layout="centered", page_title="MindWatch")

# Function to get the mood with the highest probability
def get_highest_mood(mood_list):
    mood_list = ast.literal_eval(mood_list)
    highest_mood = max(mood_list, key=lambda x: x[1])
    return highest_mood[0]

# Directory Path
directory_path = r"Results\Mood"

# List all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

# Streamlit app
st.title("Moods")

# File selection dropdown
selected_file = st.selectbox("Select a CSV file", csv_files)

# Full file path
file_path = os.path.join(directory_path, selected_file)

# Load the data
df = pd.read_csv(file_path)

# Apply the function to create a new 'Highest Mood' column
df['Highest Mood'] = df['Moods'].apply(get_highest_mood)

# Display the table with 'Frame Number' and 'Highest Mood'
st.write("Data Table:")
st.table(df[['Frame Number', 'Highest Mood']])
