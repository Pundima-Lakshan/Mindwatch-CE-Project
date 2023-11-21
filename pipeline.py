from transformers import ViltForQuestionAnswering, ViltFeatureExtractor
import cv2
import numpy as np
import torch

MODEL_PATH = "E:/violence images/vilt-b32-finetuned-vqa"

# Load the ViltForQuestionAnswering model and feature extractor
model = ViltForQuestionAnswering.from_pretrained(MODEL_PATH)
feature_extractor = ViltFeatureExtractor.from_pretrained(MODEL_PATH)

# Function to preprocess image for VQA model
def preprocess_image(image):
    # Resize image to match model's expected size
    resized_image = cv2.resize(image, (224, 224))

    # Convert image to RGB format
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Convert to NumPy array
    image_array = np.array(rgb_image)

    return image_array

# Function to ask questions about each frame in a video
def ask_questions_video(video_path, question_text):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Iterate through each frame in the video
    while True:
        # Read the frame
        ret, frame = cap.read()

        # Break the loop if no more frames
        if not ret:
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_image(frame)

        # Extract features from the preprocessed frame
        inputs = feature_extractor(images=preprocessed_frame, return_tensors="pt")

        # Ask the question about the current frame
        result = model(**inputs)

        # Print the result for the current frame
        print(result)

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    VIDEO_PATH = "E:/violenceDetection/2.mp4"
    QUESTION_TEXT = "Is those players playing boxing?"

    ask_questions_video(VIDEO_PATH, QUESTION_TEXT)
