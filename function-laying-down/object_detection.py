# Imports and Setup
# !pip install mediapipe streamlit opencv-python

# Download the image segmenter model
# !wget -O deeplabv3.tflite -q https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/dee

# Download a test image
# Import required libraries
import PIL

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

# Replace the relative path to your weight file

# Setting page layout
st.set_page_config(
    page_title="Object Detection",  # Setting page title
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
        "D:/Projects/1 CEProject/mediapipe-models/object-detection/efficientdet_lite2.tflite",
    )
    allowed_categories = st.text_input(
        "Enter allowed categories seperated by comma and space", "person, bed"
    )

# Creating main page heading
st.title("Object Detection using mediapipe")

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img, caption="Uploaded Image", use_column_width=True)


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """

    MARGIN = 10  # pixels
    ROW_SIZE = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 0, 0)  # red

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 2)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return image


# Configure the model
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

allow_list = allowed_categories.split(", ")

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=max_results,
    running_mode=VisionRunningMode.IMAGE,
    score_threshold=confidence,
    category_allowlist=allow_list,
)

with ObjectDetector.create_from_options(options) as detector:
    # Object detection if button pressed
    if st.sidebar.button("Detect Objects"):
        # Load the input image from an image file.
        mp_image = uploaded_image

        # Load the input image from a numpy array.
        numpy_image = np.array(uploaded_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        # Perform object detection on the provided single image.
        detection_result = detector.detect(mp_image)

        image_copy = np.copy(mp_image.numpy_view())
        annotated_image = visualize(image_copy, detection_result)

        with col2:
            st.image(annotated_image, caption="Detected Image", use_column_width=True)

            try:
                with st.expander("Detection Results Format"):
                    st.write(detection_result)
                with st.expander("Detection Results"):
                    for index, detection in enumerate(detection_result.detections):
                        st.write(index)
                        st.write(detection)
            except Exception as ex:
                st.write(ex)
                st.write("No image is uploaded yet!")
