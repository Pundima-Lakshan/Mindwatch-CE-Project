import csv
import os
from moviepy.editor import VideoFileClip

video_path = 'D:\\downloadd\\33.mp4'

# Extract video file name without extension
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Create CSV file name based on video file name and specify folder
output_folder = 'talking time'
output_csv_path = os.path.join(output_folder, f'{video_name}_info.csv')

# Function to determine audio status for each second
def get_audio_status(video_path, start_time, end_time):
    subclip = VideoFileClip(video_path).subclip(start_time, end_time)
    has_audio = subclip.audio is not None
    return "Audio present" if has_audio else "No audio"

# Get the total duration of the video in seconds
video_clip = VideoFileClip(video_path)
total_duration = int(video_clip.duration)
video_clip.close()

# Create and write to CSV file
with open(output_csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write header
    csv_writer.writerow(['Duration(s)', 'Audio Status'])
    
    # Write data for each second
    for second in range(total_duration):
        start_time = second
        end_time = second + 1
        audio_status = get_audio_status(video_path, start_time, end_time)
        csv_writer.writerow([start_time, audio_status])

video_clip.close()

print(f"CSV file created successfully at {output_csv_path}")

