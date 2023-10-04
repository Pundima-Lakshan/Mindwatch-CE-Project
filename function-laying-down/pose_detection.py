# Imports and Setup
# !pip install mediapipe streamlit opencv-python

# Download the image segmenter model

# Import required libraries
import PIL

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Replace the relative path to your weight file

# Setting page layout
st.set_page_config(
    page_title="Pose Detection",  # Setting page title
    layout="wide",  # Setting layout to wide
    initial_sidebar_state="expanded",  # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Image/Video Config")  # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    # Model Options
    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100
    max_results = int(st.number_input("No of max results", value=-1))
    model_path = st.text_input(
        "Absolute path to the model",
        "D:/Projects/1 CEProject/mediapipe-models/pose-detection/pose_landmarker_full.task",
    )

# Creating main page heading
st.title("Pose Detection using mediapipe")

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img, caption="Uploaded Image", use_column_width=True)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


# Configure the model
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
)

with PoseLandmarker.create_from_options(options) as landmarker:
    # Object detection if button pressed
    if st.sidebar.button("Detect pose"):
        # Load the input image from an image file.
        mp_image = uploaded_image

        # Load the input image from a numpy array.
        numpy_image = np.array(uploaded_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        # Perform object detection on the provided single image.
        pose_landmarker_result = landmarker.detect(mp_image)

        image_copy = np.copy(mp_image.numpy_view())
        annotated_image = draw_landmarks_on_image(image_copy, pose_landmarker_result)

        with col2:
            st.image(annotated_image, caption="Detected Image", use_column_width=True)

            try:
                with st.expander("Detection Results Format"):
                    st.write(pose_landmarker_result)
                with st.expander("Landmark Detection Results"):
                    for index, detection in enumerate(
                        pose_landmarker_result.pose_landmarks
                    ):
                        st.write(index)
                        st.write(detection)
                with st.expander("World Landmark Detection Results"):
                    for index, detection in enumerate(
                        pose_landmarker_result.pose_world_landmarks
                    ):
                        st.write(index)
                        st.write(detection)
            except Exception as ex:
                st.write(ex)
                st.write("No image is uploaded yet!")
