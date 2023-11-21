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
    df['TimeCategory'] = df['Time '].dt.strftime('%H:%M:%S')  # Extract only the time part
    best_options = df.groupby('TimeCategory')['Direction'].agg(lambda x: x.mode().iloc[0]).reset_index()
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
    fig = px.line(get_best_option_per_second(df), x='TimeCategory', y='Direction', title='Best Option per Second')
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
            st.write(df[['TimeCategory', 'Direction']])

        with col2:
            st.write("Best Option per Second:")
            best_options_df = get_best_option_per_second(df)
            st.write(best_options_df)

        # Additional Analysis (Example: Count of each direction)
        with col3:
            st.write("Additional Analysis:")
            direction_counts = df['Direction'].value_counts()
            st.bar_chart(direction_counts)
