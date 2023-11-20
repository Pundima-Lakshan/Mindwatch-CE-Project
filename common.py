# module to store mindwatch data

import cv2
import json
import streamlit as st
import mediapipe as mp


class MindWatch:
    config_data = None
    cap = None
    isError = None

    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    def __init__(self):
        try:
            with open("./training_config.json", "r") as file:
                self.config_data = json.load(file)
        except FileNotFoundError:
            st.error("Can't find training_config.json. Please run setup.py.")
            self.isError = True
            return
        except json.decoder.JSONDecodeError:
            st.error("training_config.json is not a valid JSON file.")
            self.isError = True
            return
        except:
            st.error("Something went wrong. Please restart the app.")
            self.isError = True
            return

        try:
            self.cap = cv2.VideoCapture(self.config_data["video_source"])
            if not self.cap.isOpened():
                raise ValueError("Can't open video source.")
        except ValueError as ve:
            st.error(str(ve))
            self.isError = True
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            self.isError = True

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results
