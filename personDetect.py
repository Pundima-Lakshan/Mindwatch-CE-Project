import cv2
import numpy as np

def is_human_in_frame(frame_path):
    # Load the YOLO model
    net = cv2.dnn.readNet("yolo v3/yolov3.weights", "yolo v3/yolov3.cfg")
    
    # Load class labels
    with open("coco.names", "r") as f:
        class_labels = f.read().strip().split("\n")
    
    # Read the frame from the specified path
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"Error: Unable to read the frame from {frame_path}")
        return False, []

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getUnconnectedOutLayersNames()
    
    # Forward pass
    outputs = net.forward(layer_names)
    
    # Initialize lists to store class IDs and class labels
    class_ids = []
    labels = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter detections by confidence threshold (adjust as needed)
            if confidence > 0.5:
                class_ids.append(class_id)
                labels.append(class_labels[class_id])
    
    # Check if "person" is in the detected labels
    if "person" in labels:
        return True, labels
    else:
        return False, labels

# Example usage with your specified frame path
frame_path = "D:/downloadd/test image/1.jpg"
human_detected, detected_labels = is_human_in_frame(frame_path)

if human_detected:
    print("A human is in the frame")
    print("Detected Labels:", detected_labels)
else:
    print("No human detected in the frame")
