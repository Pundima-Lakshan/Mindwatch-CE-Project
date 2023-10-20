from PIL import Image
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
import queue


class VideoAnalyzer:
    def __init__(self):
        self.FRAMES_TO_ANALYZE = 5
        self.output_queue = queue.Queue()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("GPU is available")
        else:
            self.device = torch.device("cpu")
            print("GPU is not available")

        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.clip_model = self.clip_model.to(self.device)

    def get_probabilities_for_frame(self, image, labels=['violent scene', 'non-violent scene']):
        inputs = self.clip_processor(
            text=labels, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs_dict = {labels[i]: probs[0][i].item()
                      for i in range(len(labels))}
        return probs_dict

    def annotate_frame(self, frame, probability_threshold):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (10, 30)
        font_scale = 1
        font_color = (0, 0, 255)
        font_thickness = 2
        text = f"Violent Probability: {probability_threshold:.2f}"
        cv2.putText(frame, text, bottom_left_corner, font,
                    font_scale, font_color, font_thickness)

    def process_video(self, input_path):
        vs = cv2.VideoCapture(input_path)

        last_scores = []
        frame_count = 0

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
            if len(last_scores) > self.FRAMES_TO_ANALYZE:
                last_scores = last_scores[1:]

            final_score = max(last_scores)
            self.output_queue.put((frame_count, final_score))

            print(f"Frame {frame_count}: Violent Probability - {final_score}")

            frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        vs.release()
        cv2.destroyAllWindows()

        while not self.output_queue.empty():
            frame_num, score = self.output_queue.get()
            print(f"Frame {frame_num}: Violent Probability - {score}")

        print("[INFO] Video processing complete.")

    def process_webcam(self):
        vs = cv2.VideoCapture(0)

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

            self.annotate_frame(frame, violent_probability)
            cv2.imshow("Webcam Feed", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        vs.release()
        cv2.destroyAllWindows()

    def main(self):
        print("Choose an option:")
        print("1. Investigate a video file")
        print("2. Examine live webcam input")
        option = input("Enter the option (1/2): ")

        if option == "1":
            input_video_path = '4.mp4'
            self.process_video(input_video_path)
        elif option == "2":
            self.process_webcam()
        else:
            print("Invalid option. Please choose 1 or 2.")


if __name__ == "__main__":
    video_analyzer = VideoAnalyzer()
    video_analyzer.main()
