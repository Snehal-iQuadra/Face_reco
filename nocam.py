import cv2
import os
import face_recognition
import time
import keyboard  # to detect key presses

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

# Initialize camera
video = cv2.VideoCapture(0)

# Define the tolerance level
tolerance = 0.4
consecutive_failures = 0
consecutive_no_faces = 0
failure_threshold = 5
check_interval = 5  # Cam Interval

# Define ANSI escape codes for colored text
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
GREY = "\033[90m"
RESET = "\033[0m"

# Display instructions
print("Press 'q' to quit.")

start_time = time.time()

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Check if it's time to capture and verify a keyframe
    if time.time() - start_time >= check_interval:
        start_time = time.time()  # Reset the timer

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_encodings:
            consecutive_no_faces += 1
            consecutive_failures += 1
            print(f"{GREY}No face detected.{RESET}")
        else:
            consecutive_no_faces = 0
            user_found = False
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(user_face_encodings, face_encoding, tolerance=tolerance)
                if True in matches:
                    user_found = True
                    break

            if user_found:
                consecutive_failures = 0
                print(f"{GREEN}{user_name} recognized.{RESET}")
            else:
                consecutive_failures += 1
                print(f"{RED}Invalid face input detected.{RESET}")

        if consecutive_no_faces >= failure_threshold:
            print(f"{RED}ERROR: {user_name} is not present in front of the camera{RESET}")
            break
        elif consecutive_failures >= failure_threshold:
            print(f"{RED}ERROR: YOU ARE NOT {user_name}{RESET}")
            break

    # Check if 'q' is pressed to quit
    if keyboard.is_pressed('q'):
        print(f"{BLUE}EXITING{RESET}")
        break

# Release handle to the webcam
video.release()
cv2.destroyAllWindows()
