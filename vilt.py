from transformers import pipeline
import cv2
import numpy as np
import torch
from PIL import Image

# Load the ViltForQuestionAnswering model and feature extractor
MODEL_PATH = "D:\\Projects\\1 CEProject\\git\\Mindwatch-CE-Project\\vilt-b32-finetuned-vqa"
model = pipeline("visual-question-answering", model=MODEL_PATH)

# Function to ask questions about each frame in a video
def ask_questions_video(video_path, questions):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    skip_count = 30
    count = 0

    # Iterate through each frame in the video
    while True:
        count += 1

        if count < skip_count:
            continue
        else:
            count = 0

        # Read the frame
        ret, frame = cap.read()

        # Break the loop if no more frames
        if not ret:
            break

        frame = Image.fromarray(frame)
        out = ask_question_image(frame, questions)

    # Release the video capture object
    cap.release()

def ask_question_image(image, questions):
    out = []

    questions_list = questions.split("? ")
    for question in questions_list:
        # print(question)
        ans = model(image=image, question=question)
        out.append(ans)
        print(ans)
        print("")

    return out

if __name__ == "__main__":
    VIDEO_PATH = "D:\\Projects\\1 CEProject\\resources\\videos\\2.mp4"
    QUESTION_TEXTS = "Is the person sleeping? What is the mood of the person?"

    ask_questions_video(VIDEO_PATH, QUESTION_TEXTS)

