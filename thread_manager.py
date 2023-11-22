import os
import json
import glob
import threading
import time
import subprocess
from Functions import Aggressive_behavior_detection
from Functions import Head_pose 
from Functions import sleeping 
from Functions import mood
from Functions import standing_on_bed

# Directory path for video files and log file
log_file_path = 'execution_log.txt'

Aggressive_behavior_detection_Class = Aggressive_behavior_detection.Aggressive_behavior_detection_Class
analyze_head_pose = Head_pose.analyze_head_pose
analyze_sleeping = sleeping.analyze_sleeping
analyze_mood = mood.analyze_mood
analyze_standing_on_bed = standing_on_bed.analyze_standing_on_bed


# Specify the path to your JSON file
json_file_path = "output.json"

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Retrieve values
directory_path = data["Input_Directory"]

aggressive_behavior_detectionJS = data["Aggressive_Behavior_Detection"]
head_pose_detectionJS = data["Head_Pose_Detection"]
laying_detectionJS = data["Laying_Detection"]
mood_detectionJS = data["Mood_Detection"]
standing_on_bed_detectionJS = data["Standing_On_Bed_Detection"]

aggressive_behavior_model = data["Aggressive_Behavior_Detection_model"]
frames_to_analyze = data["Aggressive_Behavior_Detection_frames_to_analyze"]

visiting_time_start = data["Visiting_Time_Start"]
visiting_time_end = data["Visiting_Time_End"]


def create_log_file_if_not_exists():
    if not os.path.isfile(log_file_path):
        with open(log_file_path, "w") as log_file:
            log_file.write("Execution Log:\n")


def open_file_after_execution():
    # Check if the log file exists before attempting to open it
    if os.path.isfile(log_file_path):
        # Open the log file using the default text editor
        subprocess.run(["notepad.exe", log_file_path])
    else:
        print(f"Log file '{log_file_path}' does not exist.")


def mood_detection(video_file):
    analyze_mood(video_file)
    log_execution(video_file, "mood_detection")

def head_pose_detection(video_file):
    analyze_head_pose(video_file,visiting_time_start,visiting_time_end)
    log_execution(video_file, 'head_pose_detection')

def aggressive_behavior_detection(video_file):
    Aggressive_behavior_detection_object = Aggressive_behavior_detection_Class(video_file, frames_to_analyze, aggressive_behavior_model)
    Aggressive_behavior_detection_object.process_video()
    log_execution(video_file, "aggressive_behavior_detection")

def laying_detection(video_file):
    analyze_sleeping(video_file)
    log_execution(video_file, "laying_detection")

def standing_on_bed_detection(video_file):
    analyze_standing_on_bed(video_file)
    log_execution(video_file, "standing_on_bed_detection")

def log_execution(video_file, detection):
    current_time = time.strftime("%b %d %H:%M:%S", time.localtime())
    log_message = f"{video_file} {detection} done in {current_time}\n"
    with open(log_file_path, "a") as log_file:
        log_file.write(log_message)

def is_already_executed(video_file, detection):
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            if f"{video_file} {detection} done" in line:
                return True
    return False

if __name__ == "__main__":
    create_log_file_if_not_exists()

    # List all MP4 files in the directory
    video_files = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.endswith(".mp4")
    ]

    # Create and start separate threads for each function
    threads = []
    for video_file in video_files:
        if aggressive_behavior_detectionJS and not is_already_executed(video_file, "aggressive_behavior_detection"):
            print("aggressive_behavior_detection")
            thread = threading.Thread(target=aggressive_behavior_detection, args=(video_file,))
            threads.append(thread)
        if head_pose_detectionJS and not is_already_executed(video_file, "head_pose_detection"):
            print("head_pose_detection")
            thread = threading.Thread(target=head_pose_detection, args=(video_file,))
            threads.append(thread)
        if laying_detectionJS and not is_already_executed(video_file, "laying_detection"):
            print("laying_detection")
            thread = threading.Thread(target=laying_detection, args=(video_file,))
            threads.append(thread)
        if mood_detectionJS and not is_already_executed(video_file, "mood_detection"):
            print("mood_detection")
            thread = threading.Thread(target=mood_detection, args=(video_file,))
            threads.append(thread)
        if standing_on_bed_detectionJS and not is_already_executed(video_file, "standing_on_bed_detection"):
            print("standing_on_bed_detection")
            thread = threading.Thread(target=standing_on_bed_detection, args=(video_file,))
            threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    open_file_after_execution()
    print("finished processing all videos")
