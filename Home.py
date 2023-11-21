import streamlit as st
from Pages.temp import show as temp

from Pages.head_pose import show as pose

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About"])

    if page == "Home":
        pose()
    elif page == "About":
        temp()

if __name__ == "__main__":
    main()
