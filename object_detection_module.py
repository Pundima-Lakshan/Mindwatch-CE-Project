import mediapipe as mp
import cv2
import numpy as np


class MPObjectDetection:
    def __init__(self, allow_list, model_path, max_results, confidence):
        # Configure the model
        self.BaseOptions = mp.tasks.BaseOptions
        self.ObjectDetector = mp.tasks.vision.ObjectDetector
        self.ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = self.ObjectDetectorOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            max_results=max_results,
            running_mode=self.VisionRunningMode.IMAGE,
            score_threshold=confidence,
            category_allowlist=allow_list,
        )

    def detectObjects(self, image, draw=False):
        with self.ObjectDetector.create_from_options(self.options) as detector:
            # Load the input image from a numpy array.
            numpy_image = np.array(image)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

            # Perform object detection on the provided single image.
            detection_result = detector.detect(mp_image)

            if draw:
                annotated_image = self.visualize(image_copy, detection_result)
                return detection_result, annotated_image

            return detection_result

    def visualize(self, image, detection_result) -> np.ndarray:
        """Draws bounding boxes on the input image and return it.
        Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
        Returns:
        Image with bounding boxes.
        """

        numpy_image = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        image = np.copy(mp_image.numpy_view())

        MARGIN = 10  # pixels
        ROW_SIZE = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        TEXT_COLOR = (255, 0, 0)  # red

        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 2)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + " (" + str(probability) + ")"
            text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(
                image,
                result_text,
                text_location,
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                FONT_SIZE,
                TEXT_COLOR,
                FONT_THICKNESS,
            )

        return image
