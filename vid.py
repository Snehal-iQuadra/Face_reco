import cv2
import os
import face_recognition
from datetime import datetime
import time
import subprocess

# ANSI escape sequences for terminal colors
GREEN = '\033[92m'  # Green
RED = '\033[91m'    # Red
PURPLE = '\033[95m' # Purple
RESET = '\033[0m'   # Reset color

# Function to load face encodings from a specific folder
def load_face_encodings(folder_path):
    face_encodings = []
    for file in os.listdir(folder_path):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(folder_path, file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                face_encodings.append(encodings[0])
    return face_encodings

# Function to process video file
def process_video(video_path, user_face_encodings, tolerance=0.4):
    # Record the start time
    start_time = time.time()

    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    # Calculate frame skip based on the desired interval in seconds
    frame_skip = int(2 * fps)
    print(f"Frame skip interval: {frame_skip} frames")

    # Initialize variables
    user_name = os.path.basename(os.path.dirname(user_folder))
    known_face_encodings = user_face_encodings
    success = True
    frame_count = 0

    while success:
        success, frame = video.read()
        frame_count += 1

        if frame_count % frame_skip != 0:  # Skip frames if not multiple of frame_skip
            continue

        if success:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and face encodings in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Check if multiple faces are detected
            if len(face_encodings) > 1:
                # Print error message in purple
                print(f"{PURPLE}Error: Multiple faces detected in frame {frame_count}{RESET}")
            elif len(face_encodings) == 0:
                # Print no face detected message in red
                print(f"{RED}Error: No face detected in frame {frame_count}{RESET}")

            if face_encodings:
                for face_encoding in face_encodings:
                    # Compare face encoding with known encodings
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
                    if True in matches:
                        # Print positive recognition message in green
                        print(f"{GREEN}Face recognized: {user_folder}{RESET}")
                    else:
                        # Print negative recognition message in red
                        print(f"{RED}Face not recognized{RESET}")

            # Display the frame
            cv2.imshow('Video', frame)

        # Check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    video.release()
    cv2.destroyAllWindows()

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the time taken
    print(f"Processing completed in: {elapsed_time:.2f} seconds")

# Get the user's name
user_name = input("Please enter your name: ")

# Define the path to the user's folder
user_folder = os.path.join('images', user_name)

# Check if the folder exists
if not os.path.exists(user_folder):
    print(f"No data found for user: {user_name}")
    exit()

# Load the user's face encodings
user_face_encodings = load_face_encodings(user_folder)

# Ask for video file path
video_path = input("Please enter the path to the video file: ")

# Full path to the ffmpeg executable
ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'  #path to ffmpeg.exe

# Create the output file name using the user's name
output_file_name = f"{user_name}_speed.mp4"
output_file_path = os.path.join(os.path.dirname(video_path), output_file_name)

# Speed up the video and remove audio using ffmpeg
ffmpeg_command = [
    ffmpeg_path,
    '-i', video_path,
    '-filter:v', 'setpts=PTS/2',
    '-an',  # Remove the audio
    output_file_path
]

subprocess.run(ffmpeg_command)

# Process the sped-up video file, checking every 1 second
process_video(output_file_path, user_face_encodings)
