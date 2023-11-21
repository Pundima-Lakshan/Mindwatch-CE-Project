import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import os
import re


def analyze_head_pose(video_path, min_duration, max_duration):
    # Extract the file name from the path
    file_name = os.path.basename(video_path)

    # Define the regex pattern
    pattern = r"cam1_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2}).mp4"

    # Match the pattern in the file name
    match = re.match(pattern, file_name)

    if match:
        # Extract the matched groups
        (
            start_year,
            start_month,
            start_day,
            start_hour,
            start_minute,
            start_second,
            end_year,
            end_month,
            end_day,
            end_hour,
            end_minute,
            end_second,
        ) = map(int, match.groups())

    # Combine hours, minutes, and seconds into an integer in the format HHMMSS
    start_time = start_hour * 10000 + start_minute * 100 + start_second
    end_time = end_hour * 10000 + end_minute * 100 + end_second

    if (
        (end_time > max_duration and start_time <= max_duration)
        or (end_time > min_duration and start_time <= min_duration)
        or (end_time < max_duration and start_time >= min_duration)
    ):
        return 0

    output_directory2 = "Results\Head_pose"

    mp_face_mesh = mp.solutions.face_mesh

    # Create a FaceMesh object with confidence settings
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    mp_drawing = mp.solutions.drawing_utils

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Extract the filename from the video path
    video_filename = os.path.basename(video_path)
    video_name, _ = os.path.splitext(video_filename)

    csv_filename = f"{video_name}_Head_pose_result.csv"

    # Before the while loop
    direction_data = []

    # Initialize variables to track the total duration and forward duration
    total_duration = 0
    forward_duration = 0

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            break

        start = time.time()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = face_mesh.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if (
                        idx == 33
                        or idx == 263
                        or idx == 1
                        or idx == 61
                        or idx == 291
                        or idx == 199
                    ):
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)

                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array(
                    [
                        [focal_length, 0, img_h / 2],
                        [0, focal_length, img_w / 2],
                        [0, 0, 1],
                    ]
                )

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )

                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                text = "Forward"

                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 15:
                    text = "Looking Up"
                else:
                    text = "Forward"

                duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                # Check if the duration is within the specified range
                # if min_duration <= duration <= max_duration:
                if duration >= 0:
                    time_hour = start_hour
                    time_minute = start_minute
                    time_temp = duration + start_second
                    if time_temp >= 60:
                        time_temp = time_temp - 60
                        time_minute = +1
                        if time_minute >= 60:
                            time_minute = time_minute - 1
                            time_hour = time_hour + 1
                    # Combine hours, minutes, and seconds into an integer in the format HHMMSS
                    total_time_int = time_hour * 10000 + time_minute * 100 + time_temp

                    # Convert total_time_int to a string in HH:MM:SS format
                    total_time_str = (
                        f"{time_hour:02d}:{time_minute:02d}:{int(time_temp):02d}"
                    )

                    direction_data.append((total_time_str, text))

                    total_duration += 1  # Increment the total duration

                    if text == "Forward":
                        forward_duration += 1  # Increment the forward duration

                nose_3d_projection, jacobian = cv2.projectPoints(
                    nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix
                )

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 100))

                cv2.line(image, p1, p2, (255, 0, 0), 3)

                cv2.putText(
                    image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2
                )
                cv2.putText(
                    image,
                    "x: " + str(np.round(x, 2)),
                    (500, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    image,
                    "y: " + str(np.round(x, 2)),
                    (500, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    image,
                    "z: " + str(np.round(x, 2)),
                    (500, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime

            cv2.putText(
                image,
                f"FPS: {int(fps)}",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
            )

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )

        # cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Calculate the percentage of time looking forward
    percentage_forward = (
        (forward_duration / total_duration) * 100 if total_duration > 0 else 0
    )

    print(f"Percentage of time looking forward: {percentage_forward:.2f}%")

    # Save percentage_forward to a CSV file
    # percentage_csv_filename = os.path.join(output_directory, f'{video_name}_percentage_forward.csv')
    # percentage_df = pd.DataFrame({"min duration": [min_duration],"max duration": [max_duration], "Percentage Forward": [percentage_forward]})
    # percentage_df.to_csv(percentage_csv_filename, index=False)

    # Create a DataFrame from the direction_data list
    df = pd.DataFrame(direction_data, columns=["Time ", "Direction"])

    # Construct the full path to the CSV file
    csv_file_path = os.path.join(output_directory2, csv_filename)

    # Save the DataFrame to the CSV file
    df.to_csv(csv_file_path, index=False)

    cap.release()
    cv2.destroyAllWindows()


# Example usage:
# video_path = 'D:\\downloadd\\video\\cam1_20231121120005_20231121120102.mp4'
# min_duration = 120200
# max_duration = 120204

# analyze_head_pose(video_path,min_duration,max_duration)
