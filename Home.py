import streamlit as st
from common import MindWatch

if "mw" not in st.session_state:
    st.session_state.mw = MindWatch()
    if st.session_state.mw.isError:
        st.stop()
    else:
        st.success("MindWatch initialized.")

mw = st.session_state.mw

if mw.isError:
    st.error("Something went wrong. Please restart the app.")
    st.stop()
