import PIL
import streamlit as st

import object_detection_module as odm
import pose_detection_module as pdm

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
