import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

MODEL_PATH = "D:\\Projects\\1 CEProject\\git\\Mindwatch-CE-Project\\vilt-b32-finetuned-vqa"
model = pipeline("visual-question-answering", model=MODEL_PATH)

# Function to load and display the image
def load_image(image_file):
    img = Image.open(image_file)
    st.image(img, caption='Loaded Image', use_column_width=True)

# Function to ask questions about the image
def ask_questions(IMAGE, TEXT):
    # Code to ask questions and display the answers

    out = model(question=TEXT, image=IMAGE)

    for _, item in enumerate(out):
        st.write(f"{item['answer']} : {item['score']}")

# Main Streamlit app
def main():
    st.title("Image Viewer and Question Answering")

    col1, col2 = st.columns(2)

    # Sidebar with buttons
    st.sidebar.title("Options")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with col1:
            load_image(uploaded_file)

    user_input = st.sidebar.text_input("Enter your question:", "What's in the image?")

    if st.sidebar.button("Ask Questions"):
        if not user_input == "":
            with col2:
                img = Image.open(uploaded_file)
                ask_questions(IMAGE=img, TEXT=user_input)

    st.sidebar.write("Is there a person?")
    st.sidebar.write("What is the person doing?")
    st.sidebar.write("What is the mood of the person?")


if __name__ == '__main__':
    main()
