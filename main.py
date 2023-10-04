"""
Assume there is only one person in the frame
Camera is shown from side
"""

import PIL
import streamlit as st
import numpy as np
import cv2
import math

import object_detection_module as odm
import pose_detection_module as pdm
import lines_module as lm

# Setting page layout
st.set_page_config(
    page_title="Laying on bed detection",  # Setting page title
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

    # Object Detection Model Options
    st.header("Object detection model options")
    od_confidence = (
        float(st.slider("Select Model Confidence", 25, 100, 40, key="od_conf")) / 100
    )
    od_max_results = int(
        st.number_input("No of max results", value=-1, key="od_maxres")
    )
    od_model_path = st.text_input(
        "Absolute path to the model",
        "D:/Projects/1 CEProject/mediapipe-models/object-detection/efficientdet_lite2.tflite",
        key="od_mdlpth",
    )
    od_allowed_categories = st.text_input(
        "Enter allowed categories seperated by comma and space",
        "person, bed",
        key="od_allwdcat",
    )

    # Pose Detection Model Options
    st.header("Pose detection model options")
    pd_model_path = st.text_input(
        "Absolute path to the model",
        "D:/Projects/1 CEProject/mediapipe-models/pose-detection/pose_landmarker_full.task",
        key="pd_mdlpth",
    )

# Creating main page heading
st.title("Laying on bed detection")

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img, caption="Uploaded Image", use_column_width=True)


def detectPersonState(image, detection_result) -> str:
    """
    This function is to detect whether a person is
        1. On the bed
        2. Laying on the bed
        3. Not laying on the bed

    Args
        detection_result: detection results of person and bed from mediapipe
    Return:
        list of boolean values 1, 2, 3 elements represents 3 conditions
    """

    people_detection_results = None
    bed_detection_result = None

    for detection in detection_result.detections:
        name = detection.categories[0].category_name
        if name == "bed":
            bed_detection_result = detection
        elif name == "person":
            people_detection_results = detection

    # Check whether there is a person and a bed
    if bed_detection_result is None or people_detection_results is None:
        st.write("Cannot check as bed or person is missing")
        return

    # Get person and bed bounding box results
    person_bounding_box = {
        "y": people_detection_results.bounding_box.origin_y,
        "x": people_detection_results.bounding_box.origin_x,
        "width": people_detection_results.bounding_box.width,
        "height": people_detection_results.bounding_box.height,
    }

    bed_bounding_box = {
        "y": bed_detection_result.bounding_box.origin_y,
        "x": bed_detection_result.bounding_box.origin_x,
        "width": bed_detection_result.bounding_box.width,
        "height": bed_detection_result.bounding_box.height,
    }

    # Find whether there is a person with bed range
    isThereAPerson = False

    if (person_bounding_box["y"] >= bed_bounding_box["y"]) or (
        person_bounding_box["y"] + person_bounding_box["width"]
        <= bed_bounding_box["y"] + bed_bounding_box["width"]
    ):
        isThereAPerson = True
    else:
        isThereAPerson = False

    if isThereAPerson:
        st.write("there is a person")
    else:
        st.write("there is no")

    # Find whether the person is lying down or standing up

    # RADIUS = 5
    # CIRCLE_COLOR = (0, 255, 0)
    # HEIGHT = image.shape[0]

    # start_point = bbox.origin_x, bbox.origin_y
    # end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    # cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 2)


def isWithinBox(landmark, person_bounding_box, imageWidth, imageHeight):
    x = landmark.x * imageWidth
    y = landmark.y * imageHeight

    if (
        x >= person_bounding_box["x"]
        and x <= (person_bounding_box["x"] + person_bounding_box["width"])
    ) and (
        y >= person_bounding_box["y"]
        and y <= (person_bounding_box["y"] + person_bounding_box["height"])
    ):
        return True

    return False


def markCoordinates(uploaded_image, coordinates):
    image = np.array(uploaded_image)

    point_color = (150, 255, 0)

    for coordinate in coordinates:
        x, y = int(coordinate[0]), int(coordinate[1])
        cv2.circle(image, (x, y), 5, point_color, -1)

    return image


def map_value(value, from_min=0, from_max=90, to_min=100, to_max=0):
    # Ensure the value is within the source range
    value = max(min(value, from_max), from_min)

    # Calculate the normalized value within the source range
    normalized_value = (value - from_min) / (from_max - from_min)

    # Map the normalized value to the target range
    mapped_value = to_min + normalized_value * (to_max - to_min)

    return mapped_value


od_allow_list = od_allowed_categories.split(", ")

# Object detection if button pressed
if st.sidebar.button("Detect Laying Down"):
    # Initiate Object Detection model
    od_detector = odm.MPObjectDetection(
        od_allow_list, od_model_path, od_max_results, od_confidence
    )

    # Detect objects
    od_detection_result = od_detector.detectObjects(uploaded_image)
    od_annotated_image = od_detector.visualize(uploaded_image, od_detection_result)

    # Initiate Pose Detection Model
    pd_pose_detector = pdm.MPPoseDetection(pd_model_path)

    # Detect pose
    pd_pose_landmarker_result = pd_pose_detector.detect_pose(uploaded_image)
    pd_annotated_image = pd_pose_detector.draw_landmarks_on_image(
        uploaded_image, pd_pose_landmarker_result
    )

    annotated_image = od_detector.visualize(pd_annotated_image, od_detection_result)

    with col2:
        st.image(annotated_image, caption="Detected Image", use_column_width=True)

        with st.expander("Seperate Results Annotated"):
            st.image(
                od_annotated_image, caption="Detected Image", use_column_width=True
            )
            st.image(
                pd_annotated_image, caption="Detected Image", use_column_width=True
            )

        with st.expander("Seperate Results"):
            st.write("Object detection results")
            st.write(od_detection_result)

            st.write("Pose detection results")
            st.write(pd_pose_landmarker_result)

            # ------------------------------------------------------------------------------------

        with st.expander("Updating Results"):
            people_detection_results = None
            bed_detection_result = None

            for detection in od_detection_result.detections:
                name = detection.categories[0].category_name
                if name == "bed":
                    bed_detection_result = detection
                elif name == "person":
                    people_detection_results = detection

            # Get person and bed bounding box results
            person_bounding_box = {
                "y": people_detection_results.bounding_box.origin_y,
                "x": people_detection_results.bounding_box.origin_x,
                "width": people_detection_results.bounding_box.width,
                "height": people_detection_results.bounding_box.height,
            }

            # bed_bounding_box = {
            #     "y": bed_detection_result.bounding_box.origin_y,
            #     "x": bed_detection_result.bounding_box.origin_x,
            #     "width": bed_detection_result.bounding_box.width,
            #     "height": bed_detection_result.bounding_box.height,
            # }

            copy_pose_results = pd_pose_landmarker_result
            copy_pose_landmarks_normalized = copy_pose_results.pose_landmarks[0]

            st.write("pose landmarks normalized to image")
            st.write(copy_pose_landmarks_normalized)

            remove_list_hands = [20, 22, 18, 16, 14, 13, 15, 17, 19, 21]

            copy_pose_landmarks_normalized = [
                copy_pose_landmarks_normalized[i]
                for i in range(len(copy_pose_landmarks_normalized))
                if i not in remove_list_hands
            ]

            st.write("pose landmarks hands removed normalized to image")
            st.write(copy_pose_landmarks_normalized)

            imageWidth, imageHeight = uploaded_image.size

            pose_landmarks_outofbox_removed_normalized = [
                copy_pose_landmarks_normalized[i]
                for i in range(len(copy_pose_landmarks_normalized))
                if isWithinBox(
                    copy_pose_landmarks_normalized[i],
                    person_bounding_box,
                    imageWidth,
                    imageHeight,
                )
            ]

            st.write("pose landmarks outof box removed normalized to image")
            st.write(pose_landmarks_outofbox_removed_normalized)

            pose_coordinates = []

            pose_coordinates = [
                (
                    pose_landmarks_outofbox_removed_normalized[i].x * imageWidth,
                    pose_landmarks_outofbox_removed_normalized[i].y * imageHeight,
                )
                for i in range(len(pose_landmarks_outofbox_removed_normalized))
            ]

        with st.expander("Pose Landmark Coordinates"):
            st.write(pose_coordinates)

            coordinates_marked_image = markCoordinates(
                od_annotated_image, pose_coordinates
            )

            st.image(
                coordinates_marked_image,
                caption="Detected Image",
                use_column_width=True,
            )

        with st.expander("Line Drawn Annotated"):
            # Get the best fit line
            np_array = np.array(pose_coordinates)
            gradient, intercept = lm.best_fit_line(np_array)

            line_drawn_image = lm.draw_infinite_line_on_image(
                gradient, intercept, imageHeight, imageWidth, annotated_image
            )

            st.image(line_drawn_image, caption="Detected Image", use_column_width=True)

        with st.expander("Angle between horizontal lines"):
            m1 = gradient  # best fit line gradient
            m2 = 0  # horizontal line

            # Calculate the angle between the two lines in radians
            angle_rad = math.atan(abs((m1 - m2) / (1 + m1 * m2)))

            # Convert the angle from radians to degrees
            angle_deg = math.degrees(angle_rad)

            st.write(angle_deg)

        with st.expander("Probability that the person is laying down"):
            if angle_deg > 90:
                angle_deg -= 90

            laying_prob = map_value(angle_deg)
            st.write(laying_prob)
