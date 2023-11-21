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
    df['Second'] = df['Duration (s)'].astype(int)
    best_options = df.groupby('Second')['Direction'].agg(lambda x: x.mode().iloc[0]).reset_index()
    return best_options

# Set page configuration
st.set_page_config(page_title="Head Pose Dashboard", page_icon=":pinched_fingers:", layout="wide")

# Title
st.title("Head Pose Dashboard")

# Directory Path
directory_path = r"D:\Git\Mindwatch-CE-Project\Resources\Head pose"

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
    

    best_df = get_best_option_per_second(df)

    # Create a plot using Plotly Express
    st.write("Best Option per Second:")
    fig = px.line(best_df, x='Second', y='Direction', title=' ')

    # Display the plot
    st.plotly_chart(fig)

    # Create a checkbox to toggle visibility of the table preview
    show_table_preview = st.checkbox("Show Data Preview")

    # Create two columns for Data Preview and Best Option Data Preview
    col1, col2 = st.columns(2)

    # Data Preview
    if show_table_preview:
        with col1:
            st.write("Data Preview:")
            st.write(df[['Duration (s)', 'Direction']])  # Display only 'Duration' and 'Direction' columns

        # Best Option Data Preview
        with col2:
            st.write("Best Option per Second:")
            best_options_df = get_best_option_per_second(df)
            st.write(best_options_df)