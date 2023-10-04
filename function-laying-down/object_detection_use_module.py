import PIL
import streamlit as st
import object_detection_module as odm

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

allow_list = allowed_categories.split(", ")

# Object detection if button pressed
if st.sidebar.button("Detect Objects"):
    # Initiate the model
    detector = odm.MPObjectDetection(allow_list, model_path, max_results, confidence)

    detection_result = detector.detectObjects(uploaded_image)
    annotated_image = detector.visualize(uploaded_image, detection_result)

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
