import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide", page_title="MindWatch")

def get_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

def get_best_option_per_second(df):
    df['Time '] = pd.to_datetime(df['Time '], format='%H:%M:%S')
    df['Time(HH:MM:SS)'] = df['Time '].dt.strftime('%H:%M:%S')  # Extract only the time part
    best_options = df.groupby('Time(HH:MM:SS)')['Direction'].agg(lambda x: x.mode().iloc[0]).reset_index()
    return best_options

# Title
st.title('Select CSV File for Mood Analysis')

# Directory Path
directory_path = r"Results\Head pose"

# List all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

# File selection dropdown
selected_file = st.selectbox("Select a CSV file", csv_files)

# Full file path
file_path = os.path.join(directory_path, selected_file)

# Load the data
df = get_data_from_csv(file_path)

# Display the DataFrame and Best Option DataFrame side by side
if df is not None:
    # Create a plot using Plotly Express
    st.write(" ")
    fig = px.line(get_best_option_per_second(df), x='Time(HH:MM:SS)', y='Direction', title='Head Pose per Second:')
    fig.update_layout(width=1200)
    fig.update_xaxes(type='category')  # Update the x-axis time format
    st.plotly_chart(fig)

    # Create a checkbox to toggle visibility of the table preview
    show_table_preview = st.checkbox("Show Data Preview")

    # Create two columns for Data Preview and Best Option Data Preview
    col1, col2,col3 = st.columns(3)

    # Data Preview
    if show_table_preview:
        with col1:
            st.write("Data Preview according to frame:")
            # Display only 'Time ' and 'Direction' columns, with the time part extracted
            TimeCategory=df['Time '].dt.strftime('%H:%M:%S')
             # Display only 'TimeCategory' and 'Direction' columns
            st.write(df[['Time(HH:MM:SS)', 'Direction']])

        with col2:
            st.write("Data Preview according to time:")
            best_options_df = get_best_option_per_second(df)
            st.write(best_options_df)

        # Additional Analysis (Example: Count of each direction)
        with col3:
            st.write("Additional Analysis:")
            direction_counts = df['Direction'].value_counts()
            st.bar_chart(direction_counts)


# Add another graph for data from all CSV files in the directory
st.write(" ")
all_data = pd.concat([get_data_from_csv(os.path.join(directory_path, file)).assign(File=file) for file in csv_files])

# Convert 'Time ' column to datetime
all_data['Time '] = pd.to_datetime(all_data['Time '], format='%H:%M:%S')

# Extract the time part and create 'TimeCategory' column
all_data['Time(HH:MM:SS)'] = all_data['Time '].dt.strftime('%H:%M:%S')

# Find the best option per second for aggregated data
best_options_all_data = get_best_option_per_second(all_data)

# Create a line chart for 'Best Option per Second' for aggregated data
fig_all_data = px.line(best_options_all_data, x='Time(HH:MM:SS)', y='Direction', title='Summary of Head Pose for all video:')
fig_all_data.update_layout(width=1200)  # Set the width to your desired value
fig_all_data.update_xaxes(type='category')  # Update the x-axis time format

# Show the plot
st.plotly_chart(fig_all_data)



