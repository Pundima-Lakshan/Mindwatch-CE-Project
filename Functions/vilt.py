from transformers import pipeline
import cv2
import numpy as np
import torch
from PIL import Image
import os
import csv

# Load the ViltForQuestionAnswering model and feature extractor
MODEL_PATH = "vilt-b32-finetuned-vqa"
MODEL_PATH = os.path.join(MODEL_PATH)

print(MODEL_PATH)

model = pipeline("visual-question-answering", model=MODEL_PATH)


# Function to ask questions about each frame in a video
def ask_questions_video(video_path, questions, type):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    skip_count = 30
    count = 0

    # Generate output CSV file path based on the input video file name
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv_path = f"Results\Sleeping\{video_file_name}_{type}_result.csv"

    with open(output_csv_path, mode="w", newline="") as csv_file:
        if type == "mood":
            fieldnames = ["Frame Number", "Moods"]
        elif type == "sleeping":
            fieldnames = [
                "Frame Number",
                f"{type} Probability (Yes)",
                f"{type} Probability (No)",
            ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        frame_count = 0

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

            if type == "mood":
                out = out[0]
                moods_list = []
                for _, item in enumerate(out):
                    moods_list.append([item["answer"], item["score"]])
                writer.writerow({"Frame Number": frame_count, "Moods": str(moods_list)})
                print(f"Frame {frame_count}: Moods - {moods_list}")
            elif type == "sleeping":
                out = out[0]
                for _, item in enumerate(out):
                    if item["answer"] == "yes":
                        yes_score = item["score"]
                    elif item["answer"] == "no":
                        no_score = item["score"]
                writer.writerow(
                    {
                        "Frame Number": frame_count,
                        f"{type} Probability (Yes)": yes_score,
                        f"{type} Probability (No)": no_score,
                    }
                )
                print(f"Frame {frame_count}: {type} Probability - {yes_score}")

            frame_count += 1

        # Release the video capture object
        cap.release()


def ask_question_image(image, questions):
    out = []

    questions_list = questions.split("? ")
    for question in questions_list:
        ans = model(image=image, question=question)
        out.append(ans)

    return out


if __name__ == "__main__":
    VIDEO_PATH = "D:\\Projects\\1 CEProject\\resources\\videos\\2.mp4"
    QUESTION_TEXTS = "Is the person sleeping? What is the mood of the person?"

    ask_questions_video(VIDEO_PATH, QUESTION_TEXTS, "sleeping")
