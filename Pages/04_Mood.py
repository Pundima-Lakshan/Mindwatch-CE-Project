import streamlit as st
import pandas as pd
import ast
import os

# Function to get the mood with the highest probability
def get_highest_mood(mood_list):
    mood_list = ast.literal_eval(mood_list)
    highest_mood = max(mood_list, key=lambda x: x[1])
    return highest_mood[0]

# Directory Path
directory_path = r"D:\Git\Mindwatch-CE-Project\Results\Mood"

# List all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

# Streamlit app
st.title('Select CSV File for Mood Analysis')

# File selection dropdown
selected_file = st.selectbox("Select a CSV file", csv_files)

# Full file path
file_path = os.path.join(directory_path, selected_file)

# Load the data
df = pd.read_csv(file_path)

# Apply the function to create a new 'Highest Mood' column
df['Highest Mood'] = df['Moods'].apply(get_highest_mood)

# Convert 'Frame Number' to numeric
df['Frame Number'] = pd.to_numeric(df['Frame Number'], errors='coerce')

# Create a bar chart for mood counts over time
st.title('Mood Counts over Time')
mood_counts_over_time = df.groupby(['Frame Number', 'Highest Mood']).size().unstack().fillna(0)
st.bar_chart(mood_counts_over_time)
