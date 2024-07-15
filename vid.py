import cv2
import os
import face_recognition

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
    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize variables
    user_name = os.path.basename(os.path.dirname(user_folder))
    known_face_encodings = user_face_encodings
    face_locations = []
    face_encodings = []
    process_this_frame = True
    success = True

    while success:
        success, frame = video.read()

        if process_this_frame and success:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and face encodings in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:
                for face_encoding in face_encodings:
                    # Compare face encoding with known encodings
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
                    if True in matches:
                        print(f"Face recognized: {user_folder}")
                        # Draw a green box around the recognized face
                        for (top, right, bottom, left) in face_locations:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    else:
                        print("Face not recognized.")
                        # Draw a red box around the unrecognized face
                        for (top, right, bottom, left) in face_locations:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Video', frame)

            process_this_frame = not process_this_frame

        # Check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    video.release()
    cv2.destroyAllWindows()

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

# Process the video file
process_video(video_path, user_face_encodings)
