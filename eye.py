import cv2
import mediapipe as mp
import json
import time

# Initialize MediaPipe Face Mesh with iris tracking enabled.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Start video capture.
cap = cv2.VideoCapture(0)
gaze_data = []

def get_iris_center(landmarks, iris_indices):
    """Calculate the iris center as the average position of the given iris landmarks."""
    x_sum = 0
    y_sum = 0
    for idx in iris_indices:
        lm = landmarks.landmark[idx]
        x_sum += lm.x
        y_sum += lm.y
    return x_sum / len(iris_indices), y_sum / len(iris_indices)

def get_gaze_direction(landmarks, left_iris_center):
    """
    Determine gaze direction based on the left iris center position relative to the left eye corners.
    Landmarks 33 and 133 are used as approximate left eye corners.
    """
    left_eye_left_corner = landmarks.landmark[33].x
    left_eye_right_corner = landmarks.landmark[133].x
    eye_width = left_eye_right_corner - left_eye_left_corner

    if left_iris_center[0] < left_eye_left_corner + eye_width * 0.35:
        return "Left"
    elif left_iris_center[0] > left_eye_left_corner + eye_width * 0.65:
        return "Right"
    else:
        return "Center"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image for a mirror effect and convert it to RGB.
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect face landmarks.
    results = face_mesh.process(image_rgb)
    gaze_direction = "No Face"
    iris_info = {}

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define iris indices for left and right eyes.
            left_iris_indices = list(range(468, 473))
            right_iris_indices = list(range(473, 478))
            
            # Calculate iris centers.
            left_iris_center = get_iris_center(face_landmarks, left_iris_indices)
            right_iris_center = get_iris_center(face_landmarks, right_iris_indices)

            # Determine gaze direction using the left iris center.
            gaze_direction = get_gaze_direction(face_landmarks, left_iris_center)
            
            iris_info = {
                "left_iris_center": {"x": left_iris_center[0], "y": left_iris_center[1]},
                "right_iris_center": {"x": right_iris_center[0], "y": right_iris_center[1]}
            }

            # Create a data point with the timestamp, gaze direction, and iris positions.
            data_point = {
                "timestamp": time.time(),
                "gaze_direction": gaze_direction,
                "iris_info": iris_info
            }
            gaze_data.append(data_point)
            
            # Optionally, draw the face mesh landmarks.
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            
    # Display the detected gaze direction on the frame.
    cv2.putText(image, f"Gaze: {gaze_direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gaze Tracker", image)
    
    # Exit loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the gaze and iris data to a JSON file.
with open("gaze_data.json", "w") as f:
    json.dump(gaze_data, f, indent=4)

cap.release()
cv2.destroyAllWindows()
