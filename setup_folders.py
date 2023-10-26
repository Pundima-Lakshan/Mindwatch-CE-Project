import os
import time
import numpy as np
import streamlit as st

col1, col2 = st.columns(2)
with col1:
    DATA_PATH = st.text_input("Data Path", "D:\\Projects\\1 CEProject\\git\Mindwatch-CE-Project\\MP_Data")
    actions = st.text_input("Actions", "hello, thanks, iloveyou")
    no_sequences = st.number_input("Number of Sequences", 30)
    sequence_length = st.number_input("Sequence Length", 30)

with col2:
    start_button = st.button("Start")
    state = st.empty()
    error_state = st.empty()
    action_state = st.empty()
    sequence_state = st.empty()

DATA_PATH = os.path.join(DATA_PATH)
actions = np.array(actions.split(","))

error = False

if start_button:
    state.write("Setting up folders...")
    
    for action in actions:
        if error:
            break

        action_state.write("Creating folder for {}".format(action))
        
        for sequence in range(no_sequences):
            if error:
                break

            sequence_state.write("Creating folder for sequence {}".format(sequence))
            
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except Exception as e:
                error_state.write("Error: {}".format(e))
                error = True
                pass
    
    if error:
        state.write("Error")
        action_state.write("")
        sequence_state.write("")
    else:
        state.write("Done")
        print("Done")
    
