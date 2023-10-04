from PIL import Image
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
import queue


FRAMES_TO_ANALYZE = 5  # Number of frames to analyze in each iteration

"""
FRAMES_TO_ANALYZE is a constant that determines the number of frames the code will 
analyze in each iteration while processing a video. In this code, it's set to 5, which
means that the code will analyze the most recent 5 frames of the video to make an 
assessment of whether the content is violent or non-violent. The code keeps a sliding 
window of these frames and updates the analysis as new frames are processed.
"""


output_queue = queue.Queue()  # Queue to store frame analysis results

# Check if a GPU is available and print the result (dev purpose)
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

# set the device accordingly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the CLIP model and processor
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model = clip_model.to(device)

# Function to get probabilities for a given frame
def get_probabilities_for_frame(image, labels=['violent scene', 'non-violent scene']):
    # Prepare the input for CLIP model (text and image), move to the appropriate device (CPU or GPU)
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    # Forward pass through the CLIP model
    outputs = clip_model(**inputs)

    # Extract probabilities and create a dictionary
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    probs_dict = {labels[i]: probs[0][i].item()
                  for i in range(len(labels))}
    return probs_dict

# Function to annotate a frame with a probability threshold
def annotate_frame(frame, probability_threshold):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (10, 30)
    font_scale = 1
    font_color = (0, 0, 255)  # Red color for annotations
    font_thickness = 2

    text = f"Violent Probability: {probability_threshold:.2f}"
    cv2.putText(frame, text, bottom_left_corner, font, font_scale, font_color, font_thickness)

# Function to process a video file
def process_video(input_path):
    vs = cv2.VideoCapture(input_path)  # Open the video file

    last_scores = []  # List to store the last analyzed frame scores
    frame_count = 0  # Counter for frame processing

    while True:
        (grabbed, frame) = vs.read()  # Read a frame from the video

        if not grabbed:
            break

        try:
            image = Image.fromarray(frame)
            image = image.convert("RGB")
            probs = get_probabilities_for_frame(image)
            violent_probability = probs["violent scene"]
        except Exception as e:
            violent_probability = 0

        last_scores.append(violent_probability)
        if len(last_scores) > FRAMES_TO_ANALYZE:
            last_scores = last_scores[1:]

        final_score = max(last_scores)
        output_queue.put((frame_count, final_score))

        print(f"Frame {frame_count}: Violent Probability - {final_score}")

        # annotate_frame(frame, final_score)

        # cv2.imshow("Processed Video", frame)  # Display the annotated frame

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()  # Release the video capture
    cv2.destroyAllWindows()  # Close the OpenCV windows

    # Print or process the results here
    while not output_queue.empty():
        frame_num, score = output_queue.get()
        print(f"Frame {frame_num}: Violent Probability - {score}")

    print("[INFO] Video processing complete.")

# Function to process live webcam input
def process_webcam():
    vs = cv2.VideoCapture(0)  # Open the default webcam (camera index 0)

    while True:
        (grabbed, frame) = vs.read()  # Read a frame from the webcam

        if not grabbed:
            break

        try:
            image = Image.fromarray(frame)
            image = image.convert("RGB")
            probs = get_probabilities_for_frame(image)
            violent_probability = probs["violent scene"]
        except Exception as e:
            violent_probability = 0

        annotate_frame(frame, violent_probability)  # Annotate the frame with the probability

        cv2.imshow("Webcam Feed", frame)  # Display the annotated webcam feed

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close the OpenCV windows

# Main execution starts here
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Investigate a video file")
    print("2. Examine live webcam input")
    option = input("Enter the option (1/2): ")

    if option == "1":
        input_video_path = '4.mp4'  # Provide the path to the video file you want to analyze
        process_video(input_video_path)
    elif option == "2":
        process_webcam()
    else:
        print("Invalid option. Please choose 1 or 2.")
