import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

# Set these values to show/hide certain vectors of the estimation
draw_gaze = True
draw_full_axis = True
draw_headpose = False

# Gaze Score multiplier (Higher multiplier = Gaze affects headpose estimation more)
x_score_multiplier = 10
y_score_multiplier = 10

# Threshold of how close scores should be to average between frames
threshold = .3

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Define 3D model points
face_3d = np.array([
    [0.0, 0.0, 0.0],            # Nose tip
    [0.0, -330.0, -65.0],       # Chin
    [-225.0, 170.0, -135.0],    # Left eye left corner
    [225.0, 170.0, -135.0],     # Right eye right corner
    [-150.0, -150.0, -125.0],   # Left Mouth corner
    [150.0, -150.0, -125.0]     # Right mouth corner
    ], dtype=np.float64)

# Reposition left eye corner to be the origin
leye_3d = np.array(face_3d)
leye_3d[:,0] += 225
leye_3d[:,1] -= 175
leye_3d[:,2] += 135

# Reposition right eye corner to be the origin
reye_3d = np.array(face_3d)
reye_3d[:,0] -= 225
reye_3d[:,1] -= 175
reye_3d[:,2] += 135

# Gaze scores from the previous frame
last_lx, last_rx = 0, 0
last_ly, last_ry = 0, 0

# Dictionary to count gaze directions
gaze_directions = Counter()
# Dictionary to track time spent in each direction
direction_start_time = {}
current_direction = "Straight"

def classify_gaze_direction(lx_score, ly_score, rx_score, ry_score):
    if lx_score < 0.4 and rx_score < 0.4:
        gaze_direction = "Left"
    elif lx_score > 0.6 and rx_score > 0.6:
        gaze_direction = "Right"
    elif ly_score < 0.4 and ry_score < 0.4:
        gaze_direction = "Up"
    elif ly_score > 0.6 and ry_score > 0.6:
        gaze_direction = "Down"
    else:
        gaze_direction = "Straight"
    return gaze_direction

while cap.isOpened():
    success, img = cap.read()

    # Flip + convert img from BGR to RGB
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    img.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(img)
    img.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    (img_h, img_w, img_c) = img.shape
    face_2d = []

    if not results.multi_face_landmarks:
        continue 

    for face_landmarks in results.multi_face_landmarks:
        face_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            # Convert landmark x and y to pixel coordinates
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Add the 2D coordinates to an array
            face_2d.append((x, y))
        
        # Get relevant landmarks for headpose estimation
        face_2d_head = np.array([
            face_2d[1],      # Nose
            face_2d[199],    # Chin
            face_2d[33],     # Left eye left corner
            face_2d[263],    # Right eye right corner
            face_2d[61],     # Left mouth corner
            face_2d[291]     # Right mouth corner
        ], dtype=np.float64)

        face_2d = np.asarray(face_2d)

        # Calculate left x gaze score
        if (face_2d[243,0] - face_2d[130,0]) != 0:
            lx_score = (face_2d[468,0] - face_2d[130,0]) / (face_2d[243,0] - face_2d[130,0])
            if abs(lx_score - last_lx) < threshold:
                lx_score = (lx_score + last_lx) / 2
            last_lx = lx_score

        # Calculate left y gaze score
        if (face_2d[23,1] - face_2d[27,1]) != 0:
            ly_score = (face_2d[468,1] - face_2d[27,1]) / (face_2d[23,1] - face_2d[27,1])
            if abs(ly_score - last_ly) < threshold:
                ly_score = (ly_score + last_ly) / 2
            last_ly = ly_score

        # Calculate right x gaze score
        if (face_2d[359,0] - face_2d[463,0]) != 0:
            rx_score = (face_2d[473,0] - face_2d[463,0]) / (face_2d[359,0] - face_2d[463,0])
            if abs(rx_score - last_rx) < threshold:
                rx_score = (rx_score + last_rx) / 2
            last_rx = rx_score

        # Calculate right y gaze score
        if (face_2d[253,1] - face_2d[257,1]) != 0:
            ry_score = (face_2d[473,1] - face_2d[257,1]) / (face_2d[253,1] - face_2d[257,1])
            if abs(ry_score - last_ry) < threshold:
                ry_score = (ry_score + last_ry) / 2
            last_ry = ry_score

        # The camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        # Distortion coefficients 
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Get rotational matrix from rotational vector
        l_rmat, _ = cv2.Rodrigues(l_rvec)
        r_rmat, _ = cv2.Rodrigues(r_rvec)

        # Adjust headpose vector with gaze score
        l_gaze_rvec = np.array(l_rvec)
        l_gaze_rvec[2][0] -= (lx_score-.5) * x_score_multiplier
        l_gaze_rvec[0][0] += (ly_score-.5) * y_score_multiplier

        r_gaze_rvec = np.array(r_rvec)
        r_gaze_rvec[2][0] -= (rx_score-.5) * x_score_multiplier
        r_gaze_rvec[0][0] += (ry_score-.5) * y_score_multiplier

        # Classify and display gaze direction
        gaze_direction = classify_gaze_direction(lx_score, ly_score, rx_score, ry_score)
        
        # Update time spent in current direction
        current_time = time.time()
        if current_direction != gaze_direction:
            if current_direction in direction_start_time:
                # Time spent in the previous direction
                duration = current_time - direction_start_time[current_direction]
                gaze_directions[current_direction] += duration
            # Update start time for the new direction
            direction_start_time[gaze_direction] = current_time
            current_direction = gaze_direction

        # Display the current gaze direction on the image
        cv2.putText(img, gaze_direction, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Head Pose Estimation', img)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Update time spent in the last direction
if current_direction in direction_start_time:
    duration = time.time() - direction_start_time[current_direction]
    gaze_directions[current_direction] += duration

# Function to display the pie chart
def display_pie_chart(gaze_directions):
    labels = gaze_directions.keys()
    sizes = gaze_directions.values()
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title("Gaze Directions - Time Spent")

# Display the pie chart
display_pie_chart(gaze_directions)

# Wait for 'q' to close the pie chart
while True:
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        plt.close()
        break
    plt.pause(0.1)  # Pause to allow the pie chart to be interactive
