import os
import glob
import threading
import time
import subprocess
from Aggressive_behavior_detection import Aggressive_behavior_detection_Class

# CHANGE THESE TWO ACCORDINGLY
# Directory path for video files and log file
directory_path = "E:/violenceDetection"
log_file_path = "execution_log.txt"
output_csv_path = "output.csv"  # for violence detect temp


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


def shaking_detection(video_file):
    log_execution(video_file, "shaking_detection")
    # shaking_detection logic here


def gaze_detection(video_file):
    log_execution(video_file, "gaze_detection")
    # gaze_detection logic here


def aggressive_behavior_detection(video_file):
    Aggressive_behavior_detection_object = Aggressive_behavior_detection_Class(
        video_file, 5, 16
    )
    Aggressive_behavior_detection_object.process_video()
    log_execution(video_file, "aggressive_behavior_detection")
    # aggressive_behavior_detection logic here


def laying_detection(video_file):
    log_execution(video_file, "laying_detection")
    # laying_detection logic here


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
        for detection_function in [
            shaking_detection,
            gaze_detection,
            aggressive_behavior_detection,
            laying_detection,
        ]:
            detection_name = detection_function.__name__
            if not is_already_executed(video_file, detection_name):
                thread = threading.Thread(target=detection_function, args=(video_file,))
                threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    open_file_after_execution()
