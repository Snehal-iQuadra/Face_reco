import cv2
import os
import face_recognition
import time

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
failure_threshold = 5
check_interval = 5  # Cam Interval

# Display instructions
print("Press 'q' to quit.")

start_time = time.time()

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Display the live feed
    cv2.imshow('Video', frame)

    # Check if it's time to capture and verify a keyframe
    if time.time() - start_time >= check_interval:
        start_time = time.time()  # Reset the timer

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        user_found = False
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(user_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"

            if True in matches:
                user_found = True
                name = user_name
                box_color = (0, 255, 0)
            else:
                box_color = (0, 0, 255)
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the result
        cv2.imshow('Video', frame)
        cv2.waitKey(3000)  # Display the result for 3 seconds

        if user_found:
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        if consecutive_failures >= failure_threshold:
            error_frame = cv2.putText(
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), cv2.FILLED),
                f"ERROR: YOU ARE NOT {user_name}",
                (50, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv2.imshow('Video', error_frame)
            cv2.waitKey(3000)  # box waiting time
            break

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video.release()
cv2.destroyAllWindows()
