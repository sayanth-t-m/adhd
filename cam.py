import cv2
import mediapipe as mp
import time
import math
import json
import numpy as np
from collections import deque

# Initialize MediaPipe Face Mesh with higher confidence thresholds for better accuracy
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,  # Focus on single face for better performance
    min_detection_confidence=0.7,  # Increased from 0.5
    min_tracking_confidence=0.7    # Increased from 0.5
)

# For drawing landmarks (optional)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark indices for eye contours (MediaPipe face mesh topology)
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Subset used for EAR calculation
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Iris indices for gaze estimation
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def euclidean_distance(pt1, pt2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def compute_ear(landmarks, eye_indices, image_width, image_height):
    """
    Compute the Eye Aspect Ratio (EAR) using 6 landmarks.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    pts = []
    for idx in eye_indices:
        if idx >= len(landmarks):
            return 0  # In case of index error, return 0
        lm = landmarks[idx]
        pts.append((int(lm.x * image_width), int(lm.y * image_height)))
    
    if len(pts) < 6:
        return 0
    
    vertical1 = euclidean_distance(pts[1], pts[5])
    vertical2 = euclidean_distance(pts[2], pts[4])
    horizontal = euclidean_distance(pts[0], pts[3])
    
    if horizontal < 1e-5:
        return 0  # Prevent division by zero
    
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def compute_eye_center(landmarks, eye_indices, image_width, image_height):
    """Compute the center of an eye using its landmarks."""
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
    """Compute the center of the iris using its landmarks."""
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
    """Apply moving average smoothing to reduce noise."""
    if len(data) < window_size:
        return data
    smoothed = []
    for i in range(len(data) - window_size + 1):
        window_avg = sum(data[i:i+window_size]) / window_size
        smoothed.append(window_avg)
    return smoothed

def detect_blink(ear_history, threshold, consec_frames=2):
    """Detect blinks using EAR with temporal consistency check.
       Converts the ear_history deque to a list before slicing.
    """
    ear_list = list(ear_history)  # Convert deque to list to allow slicing
    if len(ear_list) < consec_frames + 1:
        return False

    # Check for consecutive frames below threshold
    below_threshold = [ear < threshold for ear in ear_list[-consec_frames:]]
    return all(below_threshold) and ear_list[-consec_frames-1] >= threshold

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    blink_count = 0
    saccade_count = 0
    saccade_amplitudes = []
    fixation_durations = []
    total_time = 0
    fixation_start = None
    
    # Parameters (tweak as needed)
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
    start_time = time.time()
    duration = 30  # seconds
    processed_frames = 0
    dropped_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        total_time = elapsed_time
        
        # Skip frames to reduce processing load (every 2nd frame)
        if frame_count % 2 != 0:
            continue
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale_factor = 640 / width
            frame = cv2.resize(frame, (640, int(height * scale_factor)))
        
        image_height, image_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        processed_frames += 1
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                try:
                    # Compute EAR for blink detection
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
                    
                    # Compute eye centers and iris centers
                    left_center = compute_eye_center(landmarks, LEFT_EYE_CONTOUR, image_width, image_height)
                    right_center = compute_eye_center(landmarks, RIGHT_EYE_CONTOUR, image_width, image_height)
                    left_iris = compute_iris_center(landmarks, LEFT_IRIS, image_width, image_height)
                    right_iris = compute_iris_center(landmarks, RIGHT_IRIS, image_width, image_height)
                    
                    # --- Gaze Estimation ---
                    # Compute a simple gaze ratio for each eye by comparing the iris center to the eye center.
                    # We'll also compute an approximate eye width using two corner landmarks.
                    left_gaze_ratio = None
                    right_gaze_ratio = None
                    
                    if left_center and left_iris:
                        # Use landmark indices 33 and 133 as approximate left eye corners
                        pt_left = (landmarks[33].x * image_width, landmarks[33].y * image_height)
                        pt_right = (landmarks[133].x * image_width, landmarks[133].y * image_height)
                        left_eye_width = euclidean_distance(pt_left, pt_right)
                        if left_eye_width != 0:
                            left_gaze_ratio = (left_iris[0] - left_center[0]) / left_eye_width
                    
                    if right_center and right_iris:
                        # Use landmark indices 362 and 263 as approximate right eye corners
                        pt_left = (landmarks[362].x * image_width, landmarks[362].y * image_height)
                        pt_right = (landmarks[263].x * image_width, landmarks[263].y * image_height)
                        right_eye_width = euclidean_distance(pt_left, pt_right)
                        if right_eye_width != 0:
                            right_gaze_ratio = (right_iris[0] - right_center[0]) / right_eye_width
                    
                    # Average the two ratios if available
                    if left_gaze_ratio is not None and right_gaze_ratio is not None:
                        gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2.0
                    elif left_gaze_ratio is not None:
                        gaze_ratio = left_gaze_ratio
                    elif right_gaze_ratio is not None:
                        gaze_ratio = right_gaze_ratio
                    else:
                        gaze_ratio = None
                    
                    # Decide gaze direction based on the average ratio.
                    # (Depending on your camera setup, you may need to flip these directions.)
                    if gaze_ratio is not None:
                        if gaze_ratio < -0.1:
                            gaze_direction = "RIGHT"
                        elif gaze_ratio > 0.1:
                            gaze_direction = "LEFT"
                        else:
                            gaze_direction = "CENTER"
                    else:
                        gaze_direction = "UNKNOWN"
                    
                    # Add eye center to history for saccade detection (if needed)
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
                    
                    # Draw landmarks and overlay metrics
                    frame_copy = frame.copy()
                    for idx in LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR:
                        if idx < len(landmarks):
                            lm = landmarks[idx]
                            x, y = int(lm.x * image_width), int(lm.y * image_height)
                            cv2.circle(frame_copy, (x, y), 1, (0, 255, 0), -1)
                    
                    if left_iris:
                        cv2.circle(frame_copy, (int(left_iris[0]), int(left_iris[1])), 3, (255, 0, 0), -1)
                    if right_iris:
                        cv2.circle(frame_copy, (int(right_iris[0]), int(right_iris[1])), 3, (255, 0, 0), -1)
                    
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
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    dropped_frames += 1
        
        cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, image_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Eye Tracking - Press 'q' to Quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or elapsed_time > duration:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if fixation_start is not None:
        fixation_duration = time.time() - fixation_start
        if fixation_duration >= FIXATION_MIN_DURATION:
            fixation_durations.append(fixation_duration)
    
    avg_saccade_amplitude = sum(saccade_amplitudes) / len(saccade_amplitudes) if saccade_amplitudes else 0
    avg_fixation_duration = sum(fixation_durations) / len(fixation_durations) if fixation_durations else 0
    
    if len(fixation_durations) > 5:
        smoothed_fixation_durations = moving_average(fixation_durations, 3)
        avg_fixation_duration = sum(smoothed_fixation_durations) / len(smoothed_fixation_durations)
    
    frame_rate = processed_frames / total_time if total_time > 0 else 0
    drop_rate = dropped_frames / (processed_frames + dropped_frames) * 100 if (processed_frames + dropped_frames) > 0 else 0
    
    metrics = {
        "TotalCaptureTime_sec": total_time,
        "ProcessedFrames": processed_frames,
        "DroppedFrames": dropped_frames,
        "FrameRate_fps": frame_rate,
        "DropRate_percent": drop_rate,
        "BlinkMetrics": {
            "BlinkCount": blink_count,
            "BlinkRate_per_min": blink_count / total_time * 60 if total_time > 0 else 0,
        },
        "SaccadeMetrics": {
            "SaccadeCount": saccade_count,
            "SaccadeFrequency_per_min": saccade_count / total_time * 60 if total_time > 0 else 0,
            "AverageSaccadeAmplitude_pixels": avg_saccade_amplitude,
            "MaxSaccadeAmplitude_pixels": max(saccade_amplitudes) if saccade_amplitudes else 0,
            "MinSaccadeAmplitude_pixels": min(saccade_amplitudes) if saccade_amplitudes else 0,
        },
        "FixationMetrics": {
            "FixationCount": len(fixation_durations),
            "AverageFixationDuration_sec": avg_fixation_duration,
            "MaxFixationDuration_sec": max(fixation_durations) if fixation_durations else 0,
            "MinFixationDuration_sec": min(fixation_durations) if fixation_durations else 0,
            "FixationFrequency_per_min": len(fixation_durations) / total_time * 60 if total_time > 0 else 0,
        },
        "Parameters": {
            "EAR_THRESHOLD": EAR_THRESHOLD,
            "SACCADE_THRESHOLD": SACCADE_THRESHOLD,
            "EAR_CONSEC_FRAMES": EAR_CONSEC_FRAMES,
            "SACCADE_MIN_DURATION": SACCADE_MIN_DURATION,
            "FIXATION_MIN_DURATION": FIXATION_MIN_DURATION,
        }
    }
    
    with open("improved_eye_metrics.json", "w") as json_file:
        json.dump(metrics, json_file, indent=4)
    
    print("Metrics saved to improved_eye_metrics.json")
    print(f"Summary: {blink_count} blinks, {saccade_count} saccades, {len(fixation_durations)} fixations")
    print(f"Processed {processed_frames} frames at {frame_rate:.1f} fps with {drop_rate:.1f}% drop rate")

if __name__ == "__main__":
    main()
