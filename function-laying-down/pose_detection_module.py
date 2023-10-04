import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class MPPoseDetection:
    def __init__(self, model_path):
        # Configure the model
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.IMAGE,
        )

    def detect_pose(self, image, draw=False):
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            # Load the input image from a numpy array.
            numpy_image = np.array(image)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

            # Perform object detection on the provided single image.
            pose_landmarker_result = landmarker.detect(mp_image)

            if draw:
                annotated_image = self.draw_landmarks_on_image(
                    image, pose_landmarker_result
                )

                return pose_landmarker_result, annotated_image

            return pose_landmarker_result

    def draw_landmarks_on_image(self, image, detection_result):
        numpy_image = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        rgb_image = np.copy(mp_image.numpy_view())

        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in pose_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated_image
