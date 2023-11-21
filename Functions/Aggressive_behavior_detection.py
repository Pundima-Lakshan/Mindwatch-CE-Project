from PIL import Image
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
import queue
import csv  # Import the csv module
import os

class Aggressive_behavior_detection_Class:
    def __init__(self, input_path, frames_to_analyze, model):
        self.input_path = input_path
        self.frames_to_analyze = frames_to_analyze
        self.model = model

        self.output_queue = queue.Queue()

        if torch.cuda.is_available():
            print("GPU is available")
        else:
            print("GPU is not available")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if (model == 16):
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        else:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.clip_model = self.clip_model.to(self.device)

        # Generate output CSV file path based on the input video file name
        video_file_name = os.path.splitext(os.path.basename(input_path))[0]
        self.output_csv_path = f"Results/Aggressive_behavior_detection/{video_file_name}_aggressive_behavior_result.csv"
        

    def get_probabilities_for_frame(self, image, labels=['violent scene', 'non-violent scene']):
        inputs = self.clip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs_dict = {labels[i]: probs[0][i].item() for i in range(len(labels))}
        return probs_dict

    def process_video(self):
        vs = cv2.VideoCapture(self.input_path)
        last_scores = []
        frame_count = 0

        with open(self.output_csv_path, mode='w', newline='') as csv_file:
            fieldnames = ['Frame Number', 'Violence Probability']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                (grabbed, frame) = vs.read()

                if not grabbed:
                    break

                try:
                    image = Image.fromarray(frame)
                    image = image.convert("RGB")
                    probs = self.get_probabilities_for_frame(image)
                    violent_probability = probs["violent scene"]
                except Exception as e:
                    violent_probability = 0

                last_scores.append(violent_probability)
                if len(last_scores) > self.frames_to_analyze:
                    last_scores = last_scores[1:]

                final_score = max(last_scores)
                self.output_queue.put((frame_count, final_score))

                frame_count += 1

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                writer.writerow({'Frame Number': frame_count, 'Violence Probability': final_score})

        vs.release()
        cv2.destroyAllWindows()

        while not self.output_queue.empty():
            frame_num, score = self.output_queue.get()
            print(f"Frame {frame_num}: Violent Probability - {score}")

if __name__ == "__main__":
    input_video_path = '1.mp4'
    output_csv_path = 'output.csv'
    frames_to_analyze = 5
    model = 16
    processor = Aggressive_behavior_detection_Class(input_video_path, frames_to_analyze, model)
    processor.process_video()