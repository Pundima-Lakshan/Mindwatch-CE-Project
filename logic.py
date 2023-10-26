# !pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib
# !pip install mediapipe opencv-python matplotlib scikit-learn tensorflow

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import streamlit as st

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw righ

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def test_webcam(useStreamlit=False):
    cap = cv2.VideoCapture(0)

    if useStreamlit:
        latest_frame = st.empty()
        error_text = st.empty()
        stop_button = st.button("STOP")

    # while the camera is open
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        if not ret:
            if useStreamlit:
                error_text.markdown("<p style='color: red;'>Can't receive frame (stream end?). Exiting ...</p>", unsafe_allow_html=True)
            else:
                print("Can't receive frame (stream end?). Exiting ...")
            break

        if useStreamlit:
            latest_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            if stop_button:
                break

        if not useStreamlit:
            # Show to screen
            cv2.imshow('OpenCV Feed', frame)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def test_detection(useStreamlit=False):
    cap = cv2.VideoCapture(0)

    if useStreamlit:
        col1, col2 = st.columns(2)

        with col1:
            stop_button = st.button("STOP")
            latest_frame = st.empty()            
            error_text = st.empty()
        
        with col2:
            latest_result = st.empty()

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            if not ret:
                if useStreamlit:
                    error_text.markdown("<p style='color: red;'>Can't receive frame (stream end?). Exiting ...</p>", unsafe_allow_html=True)
                else:
                    print("Can't receive frame (stream end?). Exiting ...")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            if useStreamlit:
                latest_frame.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
                latest_result.write(results)
                if stop_button:
                    break
            
            # Draw landmarks
            # draw_styled_landmarks(image, results)

            if not useStreamlit:
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    
        cap.release()
        cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()

def test_detection_with_landmark_drawn(useStreamlit=False):
    cap = cv2.VideoCapture(0)

    if useStreamlit:
        col1, col2 = st.columns(2)

        with col1:
            stop_button = st.button("STOP")
            latest_frame = st.empty()            
            error_text = st.empty()
        
        with col2:
            latest_result = st.empty()

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            if not ret:
                if useStreamlit:
                    error_text.markdown("<p style='color: red;'>Can't receive frame (stream end?). Exiting ...</p>", unsafe_allow_html=True)
                else:
                    print("Can't receive frame (stream end?). Exiting ...")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            if useStreamlit:
                latest_frame.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
                latest_result.write(results)
                if stop_button:
                    break

            if not useStreamlit:
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    
        cap.release()
        cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()

# Setup Folders for Collection
def setup_folders(useStreamlit=False):

    if useStreamlit:
        col1, col2 = st.columns(2)
        with col1:
            DATA_PATH = st.text_input("Data Path", "D:\\Projects\\1 CEProject\\git\Mindwatch-CE-Project\\MP_Data")
            actions = st.text_input("Actions", "hello, thanks, iloveyou")
            no_sequences = st.number_input("Number of Sequences", 30)
            sequence_length = st.number_input("Sequence Length", 30)
        
        with col2:
            start_button = st.button("Start")
            state = st.empty()
            action_state = st.empty()
            sequence_state = st.empty()
            close_button = st.button("Close")

        DATA_PATH = os.path.join(DATA_PATH)
        actions = np.array(actions.split(","))
        
        if start_button:
            state.write("Setting up folders...")
            print("Setting up folders...")
            
            for action in actions:
                action_state.write("Creating folder for {}".format(action))
                print("Creating folder for {}".format(action))
                
                for sequence in range(no_sequences):
                    sequence_state.write("Creating folder for sequence {}".format(sequence))
                    print("Creating folder for sequence {}".format(sequence))
                    
                    try: 
                        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                    except Exception as e:
                        state.write("Error: {}".format(e))
                        print("Error: {}".format(e))
                        continue
            
            state.write("Done")
            print("Done")

        while True:
            if close_button:
                break
            time.sleep(1)

    else:
        # Path for exported data, numpy arrays
        DATA_PATH = os.path.join('D:\\Projects\\1 CEProject\\git\Mindwatch-CE-Project\\MP_Data')
        # Actions that we try to detect
        actions = np.array(['hello', 'thanks', 'iloveyou'])
        # Thirty videos worth of data
        no_sequences = 30
        # Videos are going to be 30 frames in length
        sequence_length = 30
            
        print("Setting up folders...")
        for action in actions:
            print("Creating folder for {}".format(action))
            for sequence in range(no_sequences):
                print("Creating folder for sequence {}".format(sequence))
                try: 
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                except Exception as e:
                    print("Error: {}".format(e))
                    continue
        print("Done")

# Collect Keypoint Values for Training
def collect_keypoints_for_training(actions, no_sequences, sequence_length, DATA_PATH):
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    # Draw landmarks
                    draw_styled_landmarks(image, results)                    
                    
                    # Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
    cap.release()
    cv2.destroyAllWindows()

# Prepare Data for Training

# main function
def main():
    # setup_folders()    
    # collect_keypoints_for_training(actions, no_sequences, sequence_length, DATA_PATH)    
    # test webcam
    # test_webcam_detection()    
    # Load Data & Preprocess Data    
    # Extract Keypoint Values for Training    
    # Build and Train LSTM Neural Network    
    # Make Predictions

    setup_folders()

# call main
if __name__ == "__main__":
    main()