import PIL
import streamlit as st
import pose_detection_module as pdm

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


# Object detection if button pressed
if st.sidebar.button("Detect pose"):
    # Load the input image from an image file.
    mp_image = uploaded_image

    # Initiate model
    pose_detector = pdm.MPPoseDetection(model_path)

    # Detect pose
    pose_landmarker_result = pose_detector.detect_pose(uploaded_image)
    annotated_image = pose_detector.draw_landmarks_on_image(
        mp_image, pose_landmarker_result
    )

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
