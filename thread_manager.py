import os
import glob
import threading
import multiprocessing

 
directory_path = 'E:/violenceDetection'

def shaking_detection(video_file):
	print("Processed shaking_detection for: {}" .format(video_file))


def gaze_detection(video_file):
	print("Processed gaze_detection for: {}" .format(video_file))
	

def aggressive_behavior_detection(video_file):
	print("Processed aggressive_behavior_detection for: {}" .format(video_file))


def laying_detection(video_file):
	print("Processed laying_detection for: {}" .format(video_file))


if __name__ == '__main__':
    
    # List all MP4 files in the directory
    video_files = [os.path.join(directory_path, file) for file in os.listdir(
    	directory_path) if file.endswith('.mp4')]

    # Create and start separate threads for each function
    threads = []
    for video_file in video_files:
        thread1 = multiprocessing.Process(
            target=shaking_detection, args=(video_file,))
        thread2 = multiprocessing.Process(
            target=gaze_detection, args=(video_file,))
        thread3 = multiprocessing.Process(
            target=aggressive_behavior_detection, args=(video_file,))
        thread4 = multiprocessing.Process(
            target=laying_detection, args=(video_file,))

        threads.extend([thread1, thread2, thread3, thread4])

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
