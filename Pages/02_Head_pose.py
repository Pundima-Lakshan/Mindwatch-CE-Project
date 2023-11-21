import os
import pandas as pd
import plotly.express as px
import streamlit as st

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

# Set page configuration
st.set_page_config(page_title="Head Pose Dashboard", page_icon=":pinched_fingers:", layout="wide")

# Title
st.title("Head Pose Dashboard")

# Directory Path
directory_path = r"D:\Git\Mindwatch-CE-Project\Results\Head pose"

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
    st.write("Best Option per Second:")
    fig = px.line(get_best_option_per_second(df), x='Time(HH:MM:SS)', y='Direction', title='Best Option per Second')
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
            st.write("Data Preview:")
            # Display only 'Time ' and 'Direction' columns, with the time part extracted
            TimeCategory=df['Time '].dt.strftime('%H:%M:%S')
             # Display only 'TimeCategory' and 'Direction' columns
            st.write(df[['Time(HH:MM:SS)', 'Direction']])

        with col2:
            st.write("Best Option per Second:")
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

fig_all_data = px.line(all_data, x='Time(HH:MM:SS)', y='Direction', title='Data from all CSV files')

# Increase the width of the graph
fig_all_data.update_layout(width=1200)  # Set the width to your desired value

# Update the x-axis format
fig_all_data.update_xaxes(type='category')

# Show the plot
st.plotly_chart(fig_all_data)




