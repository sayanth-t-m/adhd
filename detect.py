import cv2
import mediapipe as mp
import time
import math
import json
import numpy as np
import os
import pandas as pd
import joblib
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------- Initialization ---------------------
# Initialize MediaPipe Face Mesh with iris tracking enabled.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --------------------- Constants and Landmark Indices ---------------------
# Eye contours (for drawing and EAR)
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Subset for EAR calculation
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Iris indices (using ranges that include the iris boundary landmarks)
LEFT_IRIS = list(range(468, 473))    # e.g., 468, 469, 470, 471, 472
RIGHT_IRIS = list(range(473, 478))     # e.g., 473, 474, 475, 476, 477

# --------------------- Utility Functions ---------------------
def euclidean_distance(pt1, pt2):
    """Compute Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def compute_ear(landmarks, eye_indices, image_width, image_height):
    """
    Compute the Eye Aspect Ratio (EAR) using 6 landmarks.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    pts = []
    for idx in eye_indices:
        if idx >= len(landmarks):
            return 0  # Gracefully handle index error
        lm = landmarks[idx]
        pts.append((int(lm.x * image_width), int(lm.y * image_height)))
    if len(pts) < 6:
        return 0
    vertical1 = euclidean_distance(pts[1], pts[5])
    vertical2 = euclidean_distance(pts[2], pts[4])
    horizontal = euclidean_distance(pts[0], pts[3])
    if horizontal < 1e-5:
        return 0
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def compute_eye_center(landmarks, eye_indices, image_width, image_height):
    """Compute the centroid of the eye landmarks."""
    pts = []
    for idx in eye_indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        pts.append((lm.x * image_width, lm.y * image_height))
    if not pts:
        return None
    pts_array = np.array(pts)
    center = pts_array.mean(axis=0)
    return (center[0], center[1])

def compute_iris_center(landmarks, iris_indices, image_width, image_height):
    """Compute the centroid of the iris landmarks."""
    pts = []
    for idx in iris_indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        pts.append((lm.x * image_width, lm.y * image_height))
    if not pts:
        return None
    pts_array = np.array(pts)
    center = pts_array.mean(axis=0)
    return (center[0], center[1])

def moving_average(data, window_size):
    """Simple moving average smoothing."""
    if len(data) < window_size:
        return data
    smoothed = []
    for i in range(len(data) - window_size + 1):
        window_avg = sum(data[i:i+window_size]) / window_size
        smoothed.append(window_avg)
    return smoothed

def detect_blink(ear_history, threshold, consec_frames=2):
    """Detect a blink using temporal consistency of EAR."""
    ear_list = list(ear_history)
    if len(ear_list) < consec_frames + 1:
        return False
    below_threshold = [ear < threshold for ear in ear_list[-consec_frames:]]
    return all(below_threshold) and ear_list[-consec_frames-1] >= threshold

def get_gaze_direction(landmarks, left_iris_center, image_width):
    """
    Determine gaze direction based on the left iris center position relative to the left eye corners.
    Landmarks 33 and 133 are used as approximate left eye corners.
    """
    left_eye_left_corner = landmarks[33].x
    left_eye_right_corner = landmarks[133].x
    eye_width = left_eye_right_corner - left_eye_left_corner
    # Multiply normalized values by image_width to convert to pixel positions.
    left_corner_px = left_eye_left_corner * image_width
    right_corner_px = left_eye_right_corner * image_width
    iris_x_px = left_iris_center[0]
    if iris_x_px < left_corner_px + (right_corner_px - left_corner_px) * 0.35:
        return "Left"
    elif iris_x_px > left_corner_px + (right_corner_px - left_corner_px) * 0.65:
        return "Right"
    else:
        return "Center"

# --------------------- Eye Tracking & Gaze Metrics ---------------------
def run_eye_tracking():
    """
    Capture eye tracking metrics and gaze data from the webcam for a fixed duration,
    compute metrics, and save them to JSON files.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Set camera resolution and FPS (if available)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize metric variables.
    blink_count = 0
    saccade_count = 0
    saccade_amplitudes = []
    fixation_durations = []
    total_time = 0
    fixation_start = None

    # Parameters for detection (tune as needed)
    EAR_THRESHOLD = 0.2        
    SACCADE_THRESHOLD = 10     
    EAR_CONSEC_FRAMES = 2      
    SACCADE_MIN_DURATION = 0.02  
    FIXATION_MIN_DURATION = 0.1  

    ear_history = deque(maxlen=10)
    eye_pos_history = deque(maxlen=5)
    blink_state = False
    saccade_state = False
    saccade_start_time = None
    prev_center = None
    frame_count = 0
    processed_frames = 0
    dropped_frames = 0

    # For gaze tracking data collection.
    gaze_data = []

    start_time = time.time()
    duration = 30  # Capture duration in seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        total_time = elapsed_time

        # Process every 2nd frame for speed.
        if frame_count % 2 != 0:
            continue

        # Resize frame if needed.
        height, width = frame.shape[:2]
        if width > 640:
            scale_factor = 640 / width
            frame = cv2.resize(frame, (640, int(height * scale_factor)))
        image_height, image_width = frame.shape[:2]

        # Flip for mirror effect.
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        processed_frames += 1
        gaze_direction = "No Face"
        iris_info = {}

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # ----- Eye Metrics Computation -----
                # Compute EAR for blink detection.
                left_ear = compute_ear(landmarks, LEFT_EYE_EAR, image_width, image_height)
                right_ear = compute_ear(landmarks, RIGHT_EYE_EAR, image_width, image_height)
                ear = (left_ear + right_ear) / 2.0
                ear_history.append(ear)

                new_blink = detect_blink(ear_history, EAR_THRESHOLD, EAR_CONSEC_FRAMES)
                if new_blink and not blink_state:
                    blink_count += 1
                    blink_state = True
                elif ear > EAR_THRESHOLD * 1.2:
                    blink_state = False

                # Compute eye centers.
                left_center = compute_eye_center(landmarks, LEFT_EYE_CONTOUR, image_width, image_height)
                right_center = compute_eye_center(landmarks, RIGHT_EYE_CONTOUR, image_width, image_height)

                # ----- Gaze Metrics using Iris Tracking -----
                left_iris_center = compute_iris_center(landmarks, LEFT_IRIS, image_width, image_height)
                right_iris_center = compute_iris_center(landmarks, RIGHT_IRIS, image_width, image_height)
                if left_iris_center is not None:
                    # Use left iris center to determine gaze direction.
                    gaze_direction = get_gaze_direction(landmarks, left_iris_center, image_width)
                else:
                    gaze_direction = "Unknown"

                iris_info = {
                    "left_iris_center": {"x": left_iris_center[0] if left_iris_center else None,
                                         "y": left_iris_center[1] if left_iris_center else None},
                    "right_iris_center": {"x": right_iris_center[0] if right_iris_center else None,
                                          "y": right_iris_center[1] if right_iris_center else None}
                }

                # Record gaze data with timestamp.
                data_point = {
                    "timestamp": current_time,
                    "gaze_direction": gaze_direction,
                    "iris_info": iris_info
                }
                gaze_data.append(data_point)

                # ----- Saccade/Fixation Computation using Eye Centers -----
                if left_center is not None and right_center is not None:
                    eye_center = ((left_center[0] + right_center[0]) / 2.0,
                                  (left_center[1] + right_center[1]) / 2.0)
                    eye_pos_history.append(eye_center)
                    if len(eye_pos_history) >= 3:
                        recent_positions = list(eye_pos_history)[-3:]
                        smoothed_x = sum(p[0] for p in recent_positions) / 3
                        smoothed_y = sum(p[1] for p in recent_positions) / 3
                        smoothed_center = (smoothed_x, smoothed_y)
                        if prev_center is not None and not blink_state:
                            movement = euclidean_distance(smoothed_center, prev_center)
                            if movement > SACCADE_THRESHOLD and not saccade_state:
                                saccade_state = True
                                saccade_start_time = current_time
                                if fixation_start is not None:
                                    fixation_duration = current_time - fixation_start
                                    if fixation_duration >= FIXATION_MIN_DURATION:
                                        fixation_durations.append(fixation_duration)
                                    fixation_start = None
                            elif movement < SACCADE_THRESHOLD * 0.7 and saccade_state:
                                saccade_state = False
                                saccade_duration = current_time - saccade_start_time
                                if saccade_duration >= SACCADE_MIN_DURATION:
                                    saccade_count += 1
                                    saccade_amplitudes.append(movement)
                                fixation_start = current_time
                            elif movement < SACCADE_THRESHOLD * 0.5 and not saccade_state:
                                if fixation_start is None:
                                    fixation_start = current_time
                        else:
                            if not blink_state:
                                fixation_start = current_time
                        prev_center = smoothed_center
                    else:
                        if not blink_state:
                            fixation_start = current_time

                # ----- Drawing on Frame -----
                frame_copy = frame.copy()
                # Draw eye contours.
                for idx in LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR:
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        x, y = int(lm.x * image_width), int(lm.y * image_height)
                        cv2.circle(frame_copy, (x, y), 1, (0, 255, 0), -1)
                # Draw iris centers.
                if left_iris_center:
                    cv2.circle(frame_copy, (int(left_iris_center[0]), int(left_iris_center[1])), 3, (255, 0, 0), -1)
                if right_iris_center:
                    cv2.circle(frame_copy, (int(right_iris_center[0]), int(right_iris_center[1])), 3, (255, 0, 0), -1)
                # Optionally, draw the full face mesh.
                mp_drawing.draw_landmarks(
                    frame_copy,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

                # Overlay metrics on frame.
                cv2.putText(frame_copy, f"EAR: {ear:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame_copy, f"Blinks: {blink_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame_copy, f"Saccades: {saccade_count}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame_copy, f"Gaze: {gaze_direction}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                if blink_state:
                    cv2.putText(frame_copy, "BLINK", (image_width - 120, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif saccade_state:
                    cv2.putText(frame_copy, "SACCADE", (image_width - 120, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame_copy, "FIXATION", (image_width - 120, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                frame = frame_copy

        cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, image_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Eye Tracking & Gaze Tracker - Press 'q' to Quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or elapsed_time > duration:
            break

    cap.release()
    cv2.destroyAllWindows()
    if fixation_start is not None:
        fixation_duration = time.time() - fixation_start
        if fixation_duration >= FIXATION_MIN_DURATION:
            fixation_durations.append(fixation_duration)

    avg_saccade_amplitude = (sum(saccade_amplitudes) / len(saccade_amplitudes)) if saccade_amplitudes else 0
    avg_fixation_duration = (sum(fixation_durations) / len(fixation_durations)) if fixation_durations else 0
    if len(fixation_durations) > 5:
        smoothed_fixation_durations = moving_average(fixation_durations, 3)
        avg_fixation_duration = sum(smoothed_fixation_durations) / len(smoothed_fixation_durations)
    frame_rate = processed_frames / total_time if total_time > 0 else 0
    drop_rate = (dropped_frames / (processed_frames + dropped_frames) * 100) if (processed_frames + dropped_frames) > 0 else 0

    # Prepare a metrics dictionary.
    metrics = {
        "TotalCaptureTime_sec": total_time,
        "ProcessedFrames": processed_frames,
        "DroppedFrames": dropped_frames,
        "FrameRate_fps": frame_rate,
        "DropRate_percent": drop_rate,
        "BlinkMetrics": {
            "BlinkCount": blink_count,
            "BlinkRate_per_min": (blink_count / total_time * 60) if total_time > 0 else 0,
        },
        "SaccadeMetrics": {
            "SaccadeCount": saccade_count,
            "SaccadeFrequency_per_min": (saccade_count / total_time * 60) if total_time > 0 else 0,
            "AverageSaccadeAmplitude_pixels": avg_saccade_amplitude,
            "MaxSaccadeAmplitude_pixels": max(saccade_amplitudes) if saccade_amplitudes else 0,
            "MinSaccadeAmplitude_pixels": min(saccade_amplitudes) if saccade_amplitudes else 0,
        },
        "FixationMetrics": {
            "FixationCount": len(fixation_durations),
            "AverageFixationDuration_sec": avg_fixation_duration,
            "MaxFixationDuration_sec": max(fixation_durations) if fixation_durations else 0,
            "MinFixationDuration_sec": min(fixation_durations) if fixation_durations else 0,
            "FixationFrequency_per_min": (len(fixation_durations) / total_time * 60) if total_time > 0 else 0,
        },
        "GazeMetrics": {
            "LastGazeDirection": gaze_direction,
            "IrisInfo": iris_info
        },
        "Parameters": {
            "EAR_THRESHOLD": EAR_THRESHOLD,
            "SACCADE_THRESHOLD": SACCADE_THRESHOLD,
            "EAR_CONSEC_FRAMES": EAR_CONSEC_FRAMES,
            "SACCADE_MIN_DURATION": SACCADE_MIN_DURATION,
            "FIXATION_MIN_DURATION": FIXATION_MIN_DURATION,
        }
    }

    # Save eye tracking metrics to a JSON file.
    METRICS_FILENAME = "improved_eye_metrics.json"
    with open(METRICS_FILENAME, "w") as json_file:
        json.dump(metrics, json_file, indent=4)
    print("Metrics saved to", METRICS_FILENAME)
    print(f"Summary: {blink_count} blinks, {saccade_count} saccades, {len(fixation_durations)} fixations")
    print(f"Processed {processed_frames} frames at {frame_rate:.1f} fps with {drop_rate:.1f}% drop rate")

    # Save gaze data to a separate JSON file.
    GAZE_FILENAME = "gaze_data.json"
    with open(GAZE_FILENAME, "w") as f:
        json.dump(gaze_data, f, indent=4)
    print("Gaze data saved to", GAZE_FILENAME)

    return metrics

# --------------------- Classification ---------------------
# Configuration filenames and feature list (must match CSV headers)
MODEL_FILENAME = "adhd_model.pkl"
CSV_FILENAME = "dataset.csv"

features = [
    "SaccadeFrequency",
    "SaccadeLatency",
    "SaccadeAmplitudeVariability",
    "FixationDuration",
    "FixationScatteredness",
    "BaselinePupilDiameter",
    "PupilDilationVariability",
    "GazeDiffuseness",
    "GazeInhibition",
    "GazeConsistency",
    "BlinkRate",
    "BlinkPatternVariability",
    "SmoothPursuitAccuracy"
]

# Additional thresholds for extra scoring conditions.
RAPID_SACCADE_THRESHOLD = 30   # e.g., if SaccadeFrequency > 30 per minute
REPETITIVE_BLINK_THRESHOLD = 15  # e.g., if BlinkRate > 15 per minute

def flatten_metrics(metrics):
    """
    Flatten the nested metrics from eye tracking into a single dictionary
    containing the features required by the classifier.
    For features not computed by eye tracking, default to 0.
    """
    flat = {}
    flat["SaccadeFrequency"] = metrics.get("SaccadeMetrics", {}).get("SaccadeFrequency_per_min", 0)
    flat["SaccadeLatency"] = 0
    flat["SaccadeAmplitudeVariability"] = 0
    flat["FixationDuration"] = metrics.get("FixationMetrics", {}).get("AverageFixationDuration_sec", 0)
    flat["FixationScatteredness"] = 0
    flat["BaselinePupilDiameter"] = 0
    flat["PupilDilationVariability"] = 0
    flat["GazeDiffuseness"] = 0
    flat["GazeInhibition"] = 0
    flat["GazeConsistency"] = 0
    flat["BlinkRate"] = metrics.get("BlinkMetrics", {}).get("BlinkRate_per_min", 0)
    flat["BlinkPatternVariability"] = 0
    flat["SmoothPursuitAccuracy"] = 0
    return flat

def classify_metrics():
    """
    Load or train a Random Forest classifier using data from CSV_FILENAME.
    Then load the live metrics from the JSON file, flatten them,
    and make a prediction with additional scoring logic.
    """
    JSON_FILENAME = "improved_eye_metrics.json"
    data = pd.read_csv(CSV_FILENAME)
    print("CSV columns:", data.columns.tolist())
    missing = [feat for feat in features if feat not in data.columns]
    if missing:
        raise KeyError(f"The following required columns are missing in the CSV: {missing}")

    if os.path.exists(MODEL_FILENAME):
        clf = joblib.load(MODEL_FILENAME)
        print(f"Loaded existing model from '{MODEL_FILENAME}'.")
    else:
        X = data[features]
        y = data["Label"].map({"Neurodiverse": 1, "Neurotypical": 0})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained. Test Accuracy: {accuracy:.2f}")
        joblib.dump(clf, MODEL_FILENAME)
        print(f"Model saved to '{MODEL_FILENAME}'.")

    with open(JSON_FILENAME, "r") as f:
        live_metrics = json.load(f)
    flat_metrics = flatten_metrics(live_metrics)
    live_feature_vector = {key: [flat_metrics.get(key, 0)] for key in features}
    live_df = pd.DataFrame(live_feature_vector)
    base_prediction = clf.predict(live_df)
    base_label = "Neurodiverse" if base_prediction[0] == 1 else "Neurotypical"

    if base_label == "Neurodiverse":
        score = 1  # Base score for a Neurodiverse prediction
        rapid_eye = flat_metrics.get("SaccadeFrequency", 0) > RAPID_SACCADE_THRESHOLD
        repetitive_blink = flat_metrics.get("BlinkRate", 0) > REPETITIVE_BLINK_THRESHOLD
        if rapid_eye:
            score += 1
        if repetitive_blink:
            score += 1
        print(f"Additional conditions: Rapid Eye Movement = {rapid_eye}, Repetitive Blinking = {repetitive_blink}")
        print(f"Total Score: {score} / 3")
        final_label = "Neurodivergent" if score >= 2 else "Neurotypical"
    else:
        final_label = "Neurotypical"

    print("Final prediction:", final_label)

# --------------------- Main ---------------------
def main():
    print("Starting Eye Tracking & Gaze Tracking. Press 'q' to quit or wait for the duration to finish.")
    metrics = run_eye_tracking()
    if metrics is not None:
        print("Eye Tracking complete. Now running classification...")
        classify_metrics()

if __name__ == "__main__":
    main()
