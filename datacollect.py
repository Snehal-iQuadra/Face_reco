import cv2
import os
import time

# Initialize video capture
video = cv2.VideoCapture(0)

# Load the pre-trained face detector
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize count for images
count = 0

# Prompt user for their name and set up the directory path
nameID = input("Enter your name: ").lower()
path = os.path.join('images', nameID)

# Check if the directory already exists
if os.path.exists(path):
    print("Name already exists")
    nameID = input("Enter your name again: ").lower()
else:
    os.makedirs(path)

# Function to display a start button
def display_start_button(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Press 's' to Start"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 255, 0), 2)

# Wait for the user to press 's' to start the process
start = False
while not start:
    ret, frame = video.read()
    if not ret:
        break
    
    display_start_button(frame)
    cv2.imshow("WindowFrame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        start = True

# Capture 5 images with a delay
while count < 10:
    ret, frame = video.read()
    if not ret:
        break
    
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        name = os.path.join(path, f'{count}.jpg')
        print(f"Creating Image {count}......... {name}")
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        time.sleep(2)  # Delay of 2 seconds between captures
    
    cv2.imshow("WindowFrame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
