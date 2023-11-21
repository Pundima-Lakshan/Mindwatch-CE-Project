import streamlit as st

def show():
    st.title("Home Page")
    st.write("Welcome to the Home Page!")
    st.write("This is a simple Streamlit app with multiple pages.")
    st.button("Go to About Page", on_click=about)

def about():
    st.title("About Page")
    st.write("Welcome to the About Page!")
    st.write("This page provides information about the app.")
    st.button("Go to Home Page", on_click=show)
