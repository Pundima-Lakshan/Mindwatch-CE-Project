# app.py
import streamlit as st
from pages import home, head_pose_analysis, aggressive_behavior_analysis, climb_on_bed, shaking_analysis

# Create a dictionary to map page names to their corresponding functions
pages = {
    "Home": home.show_home,
    "Head Pose Analysis": head_pose_analysis.show_head_pose_analysis,
    "Aggressive Behavior Analysis": aggressive_behavior_analysis.show_aggressive_behavior_analysis,
    "Climb on Bed": climb_on_bed.show_climb_on_bed,
    "Shaking Analysis": shaking_analysis.show_shaking_analysis,
}

# Set the default page to "Home"
default_page = "Home"

# Create a navigation bar to switch between pages
page = st.sidebar.selectbox("", list(pages.keys()), index=list(pages.keys()).index(default_page))

# Display the selected page
pages[page]()
