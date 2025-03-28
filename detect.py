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
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading

# --------------------- MediaPipe Initialization ---------------------
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
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = list(range(468, 473))    # 468, 469, 470, 471, 472
RIGHT_IRIS = list(range(473, 478))     # 473, 474, 475, 476, 477

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
    left_corner_px = left_eye_left_corner * image_width
    right_corner_px = left_eye_right_corner * image_width
    iris_x_px = left_iris_center[0]
    if iris_x_px < left_corner_px + (right_corner_px - left_corner_px) * 0.35:
        return "Left"
    elif iris_x_px > left_corner_px + (right_corner_px - left_corner_px) * 0.65:
        return "Right"
    else:
        return "Center"

def moving_average(data, window_size):
    """Simple moving average smoothing."""
    if len(data) < window_size:
        return data
    smoothed = []
    for i in range(len(data) - window_size + 1):
        window_avg = sum(data[i:i+window_size]) / window_size
        smoothed.append(window_avg)
    return smoothed

# --------------------- Classification Functions ---------------------
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
    Returns the final label string.
    """
    JSON_FILENAME = "improved_eye_metrics.json"
    data = pd.read_csv(CSV_FILENAME)
    missing = [feat for feat in features if feat not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns in the CSV: {missing}")

    if os.path.exists(MODEL_FILENAME):
        clf = joblib.load(MODEL_FILENAME)
        print(f"Loaded model from '{MODEL_FILENAME}'.")
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
        # If score >= 2, label as ADHD; else, Neurotypical.
        final_label = ("ADHD" if score >= 2 else "Neurotypical")
    else:
        final_label = "Neurotypical"
    print("Final prediction:", final_label)
    return final_label

# --------------------- GUI Application ---------------------
class EyeTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Tracking & Gaze Tracker")
        
        # Video display label
        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=10)
        
        # Buttons frame
        control_frame = tk.Frame(root)
        control_frame.pack(padx=10, pady=5)
        
        self.start_button = tk.Button(control_frame, text="Start Eye Tracking", command=self.start_tracking)
        self.start_button.grid(row=0, column=0, padx=5)
        self.stop_button = tk.Button(control_frame, text="Stop", command=self.stop_tracking, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        self.upload_button = tk.Button(control_frame, text="Upload Video", command=self.upload_video)
        self.upload_button.grid(row=0, column=2, padx=5)
        self.classify_button = tk.Button(control_frame, text="Run Classification", command=self.run_classification, state=tk.DISABLED)
        self.classify_button.grid(row=0, column=3, padx=5)
        
        # Log output
        self.log_text = tk.Text(root, height=8, width=100)
        self.log_text.pack(padx=10, pady=10)
        
        # Progress bar (hidden by default)
        self.progress = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=400)
        self.progress.pack(padx=10, pady=5)
        self.progress["value"] = 0
        self.progress.pack_forget()
        
        # Video capture and control variables
        self.cap = None
        self.running = False
        self.video_thread = None
        
        # Metrics and tracking variables (for live tracking)
        self.blink_count = 0
        self.saccade_count = 0
        self.saccade_amplitudes = []
        self.fixation_durations = []
        self.gaze_data = []
        self.ear_history = deque(maxlen=10)
        self.eye_pos_history = deque(maxlen=5)
        self.blink_state = False
        self.saccade_state = False
        self.saccade_start_time = None
        self.fixation_start = None
        self.prev_center = None
        self.frame_count = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        
        # Parameters for detection
        self.EAR_THRESHOLD = 0.2        
        self.SACCADE_THRESHOLD = 10     
        self.EAR_CONSEC_FRAMES = 2      
        self.SACCADE_MIN_DURATION = 0.02  
        self.FIXATION_MIN_DURATION = 0.1  
        
        self.start_time = None
        self.duration = 30  # seconds

        # To hold the latest computed gaze direction and iris info
        self.last_gaze_direction = "No Face"
        self.last_iris_info = {}
        
        # To indicate which mode is active: "live" or "video"
        self.mode = "live"

    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    # --------------------- Live Camera Methods ---------------------
    def start_tracking(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return
        # Set camera resolution and FPS if possible
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        self.mode = "live"
        self.start_time = time.time()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.classify_button.config(state=tk.DISABLED)
        self.log("Starting live eye tracking...")
        self.video_loop()

    def stop_tracking(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.classify_button.config(state=tk.NORMAL)
        self.log("Stopped eye tracking.")
        
        # Finalize fixation durations if needed
        if self.fixation_start is not None:
            fixation_duration = time.time() - self.fixation_start
            if fixation_duration >= self.FIXATION_MIN_DURATION:
                self.fixation_durations.append(fixation_duration)
        
        # Compute additional metrics
        total_time = time.time() - self.start_time
        avg_saccade_amplitude = (sum(self.saccade_amplitudes) / len(self.saccade_amplitudes)) if self.saccade_amplitudes else 0
        avg_fixation_duration = (sum(self.fixation_durations) / len(self.fixation_durations)) if self.fixation_durations else 0
        if len(self.fixation_durations) > 5:
            smoothed_fixation_durations = moving_average(self.fixation_durations, 3)
            avg_fixation_duration = sum(smoothed_fixation_durations) / len(smoothed_fixation_durations)
        frame_rate = self.processed_frames / total_time if total_time > 0 else 0
        drop_rate = (self.dropped_frames / (self.processed_frames + self.dropped_frames) * 100) if (self.processed_frames + self.dropped_frames) > 0 else 0

        # Prepare a metrics dictionary similar to the original code.
        metrics = {
            "TotalCaptureTime_sec": total_time,
            "ProcessedFrames": self.processed_frames,
            "DroppedFrames": self.dropped_frames,
            "FrameRate_fps": frame_rate,
            "DropRate_percent": drop_rate,
            "BlinkMetrics": {
                "BlinkCount": self.blink_count,
                "BlinkRate_per_min": (self.blink_count / total_time * 60) if total_time > 0 else 0,
            },
            "SaccadeMetrics": {
                "SaccadeCount": self.saccade_count,
                "SaccadeFrequency_per_min": (self.saccade_count / total_time * 60) if total_time > 0 else 0,
                "AverageSaccadeAmplitude_pixels": avg_saccade_amplitude,
                "MaxSaccadeAmplitude_pixels": max(self.saccade_amplitudes) if self.saccade_amplitudes else 0,
                "MinSaccadeAmplitude_pixels": min(self.saccade_amplitudes) if self.saccade_amplitudes else 0,
            },
            "FixationMetrics": {
                "FixationCount": len(self.fixation_durations),
                "AverageFixationDuration_sec": avg_fixation_duration,
                "MaxFixationDuration_sec": max(self.fixation_durations) if self.fixation_durations else 0,
                "MinFixationDuration_sec": min(self.fixation_durations) if self.fixation_durations else 0,
                "FixationFrequency_per_min": (len(self.fixation_durations) / total_time * 60) if total_time > 0 else 0,
            },
            "GazeMetrics": {
                "LastGazeDirection": self.last_gaze_direction,
                "IrisInfo": self.last_iris_info
            },
            "Parameters": {
                "EAR_THRESHOLD": self.EAR_THRESHOLD,
                "SACCADE_THRESHOLD": self.SACCADE_THRESHOLD,
                "EAR_CONSEC_FRAMES": self.EAR_CONSEC_FRAMES,
                "SACCADE_MIN_DURATION": self.SACCADE_MIN_DURATION,
                "FIXATION_MIN_DURATION": self.FIXATION_MIN_DURATION,
            }
        }
        # Save metrics to JSON file.
        with open("improved_eye_metrics.json", "w") as json_file:
            json.dump(metrics, json_file, indent=4)
        self.log("Metrics saved to 'improved_eye_metrics.json'.")
        self.log(f"Summary: {self.blink_count} blinks, {self.saccade_count} saccades, {len(self.fixation_durations)} fixations")
        self.log(f"Processed {self.processed_frames} frames at {frame_rate:.1f} fps with {drop_rate:.1f}% drop rate")

        # Also save gaze data
        with open("gaze_data.json", "w") as f:
            json.dump(self.gaze_data, f, indent=4)
        self.log("Gaze data saved to 'gaze_data.json'.")

    def video_loop(self):
        if self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.log("Failed to capture frame.")
                self.root.after(10, self.video_loop)
                return
            
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Process every 2nd frame for speed.
            if self.frame_count % 2 != 0:
                self.root.after(1, self.video_loop)
                return

            self.processed_frames += 1
            
            # Resize if needed
            height, width = frame.shape[:2]
            if width > 640:
                scale_factor = 640 / width
                frame = cv2.resize(frame, (640, int(height * scale_factor)))
            image_height, image_width = frame.shape[:2]

            # Flip frame for mirror effect.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            gaze_direction = "No Face"
            iris_info = {}
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    # ----- Eye Metrics Computation -----
                    left_ear = compute_ear(landmarks, LEFT_EYE_EAR, image_width, image_height)
                    right_ear = compute_ear(landmarks, RIGHT_EYE_EAR, image_width, image_height)
                    ear = (left_ear + right_ear) / 2.0
                    self.ear_history.append(ear)

                    new_blink = detect_blink(self.ear_history, self.EAR_THRESHOLD, self.EAR_CONSEC_FRAMES)
                    if new_blink and not self.blink_state:
                        self.blink_count += 1
                        self.blink_state = True
                    elif ear > self.EAR_THRESHOLD * 1.2:
                        self.blink_state = False

                    # ----- Compute Eye Centers and Iris Data -----
                    left_center = compute_eye_center(landmarks, LEFT_EYE_CONTOUR, image_width, image_height)
                    right_center = compute_eye_center(landmarks, RIGHT_EYE_CONTOUR, image_width, image_height)
                    
                    left_iris_center = compute_iris_center(landmarks, LEFT_IRIS, image_width, image_height)
                    right_iris_center = compute_iris_center(landmarks, RIGHT_IRIS, image_width, image_height)
                    if left_iris_center is not None:
                        gaze_direction = get_gaze_direction(landmarks, left_iris_center, image_width)
                    else:
                        gaze_direction = "Unknown"
                    iris_info = {
                        "left_iris_center": {"x": left_iris_center[0] if left_iris_center else None,
                                               "y": left_iris_center[1] if left_iris_center else None},
                        "right_iris_center": {"x": right_iris_center[0] if right_iris_center else None,
                                                "y": right_iris_center[1] if right_iris_center else None}
                    }
                    self.last_gaze_direction = gaze_direction
                    self.last_iris_info = iris_info

                    # Record gaze data with timestamp.
                    data_point = {
                        "timestamp": current_time,
                        "gaze_direction": gaze_direction,
                        "iris_info": iris_info
                    }
                    self.gaze_data.append(data_point)

                    # ----- Saccade/Fixation Computation using Eye Centers -----
                    if left_center is not None and right_center is not None:
                        eye_center = ((left_center[0] + right_center[0]) / 2.0,
                                      (left_center[1] + right_center[1]) / 2.0)
                        self.eye_pos_history.append(eye_center)
                        if len(self.eye_pos_history) >= 3:
                            recent_positions = list(self.eye_pos_history)[-3:]
                            smoothed_x = sum(p[0] for p in recent_positions) / 3
                            smoothed_y = sum(p[1] for p in recent_positions) / 3
                            smoothed_center = (smoothed_x, smoothed_y)
                            if self.prev_center is not None and not self.blink_state:
                                movement = euclidean_distance(smoothed_center, self.prev_center)
                                if movement > self.SACCADE_THRESHOLD and not self.saccade_state:
                                    self.saccade_state = True
                                    self.saccade_start_time = current_time
                                    if self.fixation_start is not None:
                                        fixation_duration = current_time - self.fixation_start
                                        if fixation_duration >= self.FIXATION_MIN_DURATION:
                                            self.fixation_durations.append(fixation_duration)
                                        self.fixation_start = None
                                elif movement < self.SACCADE_THRESHOLD * 0.7 and self.saccade_state:
                                    self.saccade_state = False
                                    saccade_duration = current_time - self.saccade_start_time
                                    if saccade_duration >= self.SACCADE_MIN_DURATION:
                                        self.saccade_count += 1
                                        self.saccade_amplitudes.append(movement)
                                    self.fixation_start = current_time
                                elif movement < self.SACCADE_THRESHOLD * 0.5 and not self.saccade_state:
                                    if self.fixation_start is None:
                                        self.fixation_start = current_time
                            else:
                                if not self.blink_state:
                                    self.fixation_start = current_time
                            self.prev_center = smoothed_center
                        else:
                            if not self.blink_state:
                                self.fixation_start = current_time

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
                    mp_drawing.draw_landmarks(
                        frame_copy,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
                    cv2.putText(frame_copy, f"EAR: {ear:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"Blinks: {self.blink_count}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"Saccades: {self.saccade_count}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"Gaze: {gaze_direction}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    if self.blink_state:
                        cv2.putText(frame_copy, "BLINK", (image_width - 120, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif self.saccade_state:
                        cv2.putText(frame_copy, "SACCADE", (image_width - 120, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame_copy, "FIXATION", (image_width - 120, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    frame = frame_copy

            cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, image_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Convert frame for Tkinter display
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            self.root.after(10, self.video_loop)
        else:
            # If not running, clear the video display.
            self.video_label.configure(image='')

    # --------------------- Video Upload Methods with Progress Bar ---------------------
    def upload_video(self):
        video_path = filedialog.askopenfilename(title="Select Video File", 
                                                filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not video_path:
            return
        self.log(f"Processing video: {video_path}")
        self.mode = "video"
        # Show the progress bar and reset its value.
        self.progress["value"] = 0
        self.progress.pack(padx=10, pady=5)
        self.upload_button.config(state=tk.DISABLED)
        # Process video in a separate thread.
        self.video_thread = threading.Thread(target=self.process_video, args=(video_path,))
        self.video_thread.start()

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log("Error: Could not open video file.")
            self.root.after(0, lambda: self.progress.pack_forget())
            self.root.after(0, lambda: self.upload_button.config(state=tk.NORMAL))
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            # (Optional) Process the frame similarly to live tracking, if needed.
            # Here, we are not displaying the video, only simulating processing.
            time.sleep(0.03)  # Simulate processing delay.
            progress_percent = (current_frame / total_frames) * 100
            # Schedule progress update on the main thread.
            self.root.after(0, lambda p=progress_percent: self.progress.configure(value=p))
        cap.release()
        # Hide progress bar and re-enable the upload button.
        self.root.after(0, lambda: self.progress.pack_forget())
        self.root.after(0, lambda: self.upload_button.config(state=tk.NORMAL))
        self.log("Finished processing video.")
        # Automatically run classification after video processing.
        result = classify_metrics()
        self.log(f"Classification complete: {result}")
        self.root.after(0, lambda: messagebox.showinfo("Classification Result", f"Final prediction: {result}"))

    # --------------------- Classification Button ---------------------
    def run_classification(self):
        self.log("Running classification...")
        try:
            result = classify_metrics()
            self.log(f"Classification complete: {result}")
            messagebox.showinfo("Classification Result", f"Final prediction: {result}")
        except Exception as e:
            self.log(f"Classification error: {e}")
            messagebox.showerror("Error", f"Classification error: {e}")

# --------------------- Main ---------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackerGUI(root)
    root.mainloop()
