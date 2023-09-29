import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh

# Create a FaceMesh object with confidence settings
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Open the MP4 video file
cap = cv2.VideoCapture('D:\\downloadd\\Neck and Shoulder Pain Relief in Seconds.mp4')

# Before the while loop
direction_data = []

# Define the duration range (in seconds)
min_duration = 30  # Minimum duration
max_duration = 50  # Maximum duration

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
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)

            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

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
            if min_duration <= duration <= max_duration:
                direction_data.append((duration, text))

                total_duration += 1  # Increment the total duration

                if text == "Forward":
                    forward_duration += 1  # Increment the forward duration

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 100))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(x, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(x, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Calculate the percentage of time looking forward
percentage_forward = (forward_duration / total_duration) * 100 if total_duration > 0 else 0

print(f"Percentage of time looking forward: {percentage_forward:.2f}%")

# Create a DataFrame from the direction_data list
df = pd.DataFrame(direction_data, columns=["Duration (s)", "Direction"])

# Save the DataFrame to a CSV file
#df.to_csv('direction_data.csv', index=False)


cap.release()
cv2.destroyAllWindows()
