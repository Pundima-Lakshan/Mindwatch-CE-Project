# head_pose_analysis.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

def show_head_pose_analysis():
    st.title("Head Pose Analysis Page")

    # Get the list of CSV files in the "data\head_pose_analysis\graph_data" folder
    graph_data_folder = "data/head_pose_analysis/graph_data"
    graph_csv_files = [f for f in os.listdir(graph_data_folder) if f.endswith(".csv")]

    # Get the list of CSV files in the "data\head_pose_analysis\percentage" folder
    percentage_folder = "data/head_pose_analysis/percentage"
    percentage_csv_files = [f for f in os.listdir(percentage_folder) if f.endswith(".csv")]
    
    # Allow the user to select a CSV file
    selected_file = st.selectbox("Select a CSV file", graph_csv_files)

    # Load and preview the selected CSV file
    if selected_file:
        selected_file_path = os.path.join(graph_data_folder, selected_file)
        data = pd.read_csv(selected_file_path)
        st.dataframe(data)

        # Create a bar chart
        fig, ax = plt.subplots()

        categories = data['Direction'].unique()
        colors = {'Forward': 'blue', 'Looking Up': 'green', 'Looking Left': 'red', 'Looking Right': 'purple'}

        # Sequential index for x values
        x = range(len(data))

        for category in categories:
            behavior = [1 if cat == category else 0 for cat in data['Direction']]
            ax.bar(x, behavior, label=category, color=colors[category])

        # Customize the x-axis ticks to show every 5 seconds
        x_ticks = range(0, len(data), 58)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x:.2f}' for x in data['Duration (s)'][::58]])  # Show every 5th label with 2 decimal places  # Show every 5th label
        ax.set_xlabel('Duration (s)')
        ax.set_ylabel('Behavior (0/1)')
        ax.set_title('Head Pose Analysis')
        ax.legend()

        # Show the bar chart
        st.pyplot(fig)

    # Add a blank space between selections
    st.text(" ")
    st.text(" ")
    st.text(" ")

    # Allow the user to select a CSV file for "percentage"
    selected_percentage_file = st.selectbox("Select a CSV file for percentage data", percentage_csv_files)

    # Load and preview the selected CSV file for "percentage"
    if selected_percentage_file:
        selected_file_path = os.path.join(percentage_folder, selected_percentage_file)
        data = pd.read_csv(selected_file_path)
        st.dataframe(data)
        
        # Create a pie chart to display the "Percentage Forward" value
        percentage_value = data.iloc[0]['Percentage Forward']
        st.write(f"Percentage Forward: {percentage_value:.4f}")
        
        labels = ["Percentage Forward", "Remaining"]
        sizes = [percentage_value, 100 - percentage_value]
        colors = ['green', 'red']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig)

# Run the Streamlit app
if __name__ == "__main__":
    show_head_pose_analysis()
