import cv2
import numpy as np
import dlib
import time
import statistics

class ADHDMovementAnalyzer:
    def __init__(self, 
                 predictor_path='shape_predictor_68_face_landmarks.dat',
                 eye_movement_threshold=3, 
                 eye_rapid_threshold=10,
                 face_movement_threshold=10,          # Increased base threshold
                 rapid_face_multiplier=3,             # Increased multiplier for rapid face movements
                 initial_blink_ear_threshold=0.2,
                 tracking_duration=60,
                 calibration_frames=30,
                 face_calibration_frames=30):         # New parameter for face calibration
        """
        Initialize ADHD Movement Analyzer with calibration for face movements.
        """
        # Face and landmark detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Movement tracking parameters
        self.eye_movement_threshold = eye_movement_threshold
        self.eye_rapid_threshold = eye_rapid_threshold
        self.face_movement_threshold = face_movement_threshold
        self.rapid_face_multiplier = rapid_face_multiplier
        self.blink_ear_threshold = initial_blink_ear_threshold
        self.tracking_duration = tracking_duration
        self.calibration_frames = calibration_frames
        self.face_calibration_frames = face_calibration_frames
        
        # Calibration variables
        self.calibration_counter = 0
        self.ear_baseline_list = []
        self.face_baseline_list = []
        self.face_calibration_counter = 0
        
        # Movement history tracking
        self.eye_movement_history = []
        self.face_movement_history = []
        self.blink_history = []
        
        # Previous state tracking
        self.previous_face_center = None
        self.previous_eye_center = None

    def _calculate_ear(self, eye_points):
        # Calculate the Eye Aspect Ratio (EAR)
        A = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - 
                           np.array([eye_points[5].x, eye_points[5].y]))
        B = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - 
                           np.array([eye_points[4].x, eye_points[4].y]))
        C = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - 
                           np.array([eye_points[3].x, eye_points[3].y]))
        ear = (A + B) / (2.0 * C)
        return ear

    def _detect_iris(self, eye_region):
        # Iris detection using HoughCircles with fallback to thresholding.
        eye_blur = cv2.GaussianBlur(eye_region, (5, 5), 0)
        circles = cv2.HoughCircles(eye_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                   param1=50, param2=15, minRadius=3, maxRadius=15)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                return (i[0], i[1])
        _, thresh = cv2.threshold(eye_blur, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        return None

    def analyze_movement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        movement_results = {
            'eye_movement': None,
            'face_movement': None,
            'blink': None
        }
        
        if faces:
            face_rect = faces[0]
            landmarks = self.predictor(gray, face_rect)
            face_center = (
                (face_rect.left() + face_rect.right()) // 2,
                (face_rect.top() + face_rect.bottom()) // 2
            )
            
            # Calculate face movement if previous center exists
            if self.previous_face_center is not None:
                dx = face_center[0] - self.previous_face_center[0]
                dy = face_center[1] - self.previous_face_center[1]
                movement_magnitude = np.sqrt(dx**2 + dy**2)
                
                # Calibration for face movement: collect baseline data
                if self.face_calibration_counter < self.face_calibration_frames:
                    self.face_baseline_list.append(movement_magnitude)
                    self.face_calibration_counter += 1
                else:
                    # Use baseline average to adjust threshold if desired
                    baseline_face_move = statistics.mean(self.face_baseline_list)
                    # Only consider movements significantly above the baseline:
                    if movement_magnitude < (baseline_face_move * 1.5):
                        movement_magnitude = 0

                movement_results['face_movement'] = {
                    'is_moving': movement_magnitude > self.face_movement_threshold,
                    'movement_magnitude': movement_magnitude,
                    'is_rapid': movement_magnitude > (self.face_movement_threshold * self.rapid_face_multiplier)
                }
                self.face_movement_history.append(movement_results['face_movement'])
            self.previous_face_center = face_center
            
            # Eye analysis
            left_eye_points = [landmarks.part(i) for i in range(36, 42)]
            right_eye_points = [landmarks.part(i) for i in range(42, 48)]
            left_ear = self._calculate_ear(left_eye_points)
            right_ear = self._calculate_ear(right_eye_points)
            ear = (left_ear + right_ear) / 2.0

            if self.calibration_counter < self.calibration_frames:
                self.ear_baseline_list.append(ear)
                self.calibration_counter += 1
                if self.calibration_counter == self.calibration_frames:
                    baseline = statistics.mean(self.ear_baseline_list)
                    self.blink_ear_threshold = baseline * 0.75  # Adjust multiplier as needed

            is_blink = ear < self.blink_ear_threshold
            movement_results['blink'] = {
                'is_blinking': is_blink,
                'ear': ear
            }
            self.blink_history.append(movement_results['blink'])
            
            def get_eye_region(eye_points):
                x_coords = [p.x for p in eye_points]
                y_coords = [p.y for p in eye_points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                return (x_min, y_min, x_max - x_min, y_max - y_min)
            
            left_eye_region = get_eye_region(left_eye_points)
            right_eye_region = get_eye_region(right_eye_points)
            
            left_eye_crop = gray[
                left_eye_region[1]:left_eye_region[1]+left_eye_region[3],
                left_eye_region[0]:left_eye_region[0]+left_eye_region[2]
            ]
            right_eye_crop = gray[
                right_eye_region[1]:right_eye_region[1]+right_eye_region[3],
                right_eye_region[0]:right_eye_region[0]+right_eye_region[2]
            ]
            
            left_eye_center = np.mean([(p.x, p.y) for p in left_eye_points], axis=0)
            right_eye_center = np.mean([(p.x, p.y) for p in right_eye_points], axis=0)
            eye_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                          (left_eye_center[1] + right_eye_center[1]) / 2)
            
            if self.previous_eye_center is not None:
                ex = eye_center[0] - self.previous_eye_center[0]
                ey = eye_center[1] - self.previous_eye_center[1]
                eye_magnitude = np.sqrt(ex**2 + ey**2)
                movement_results['eye_movement'] = {
                    'is_moving': eye_magnitude > self.eye_movement_threshold,
                    'movement_magnitude': eye_magnitude,
                    'is_rapid': eye_magnitude > self.eye_rapid_threshold
                }
                self.eye_movement_history.append(movement_results['eye_movement'])
            self.previous_eye_center = eye_center
            
            # Optionally, iris detection can be visualized
            iris_left = self._detect_iris(left_eye_crop)
            iris_right = self._detect_iris(right_eye_crop)
            
        return movement_results

    def _calculate_adhd_score(self):
        # Total counts from movement history
        total_face_movements = sum(1 for m in self.face_movement_history if m['is_moving'])
        rapid_face_movements = sum(1 for m in self.face_movement_history if m['is_rapid'])
        total_eye_movements = sum(1 for m in self.eye_movement_history if m['is_moving'])
        rapid_eye_movements = sum(1 for m in self.eye_movement_history if m['is_rapid'])
        total_blinks = sum(1 for b in self.blink_history if b['is_blinking'])
        
        # Calculate blink rate per minute
        blink_rate = (total_blinks * 60) / self.tracking_duration if self.tracking_duration > 0 else 0
        
        # Compute proportions of rapid movements (if total movements exist)
        rapid_face_proportion = (rapid_face_movements / total_face_movements) if total_face_movements > 0 else 0
        rapid_eye_proportion = (rapid_eye_movements / total_eye_movements) if total_eye_movements > 0 else 0

        # ADHD indicators based on criteria:
        # Blink rate must be above 25 per minute
        blink_indicator = 1 if blink_rate > 25 else 0
        # Rapid movements must be above 50% of total movements
        rapid_face_indicator = 1 if rapid_face_proportion > 0.5 else 0
        rapid_eye_indicator = 1 if rapid_eye_proportion > 0.5 else 0

        # Overall ADHD score: average of the three indicators (scaled to 100)
        total_indicator = (blink_indicator + rapid_face_indicator + rapid_eye_indicator) / 3 * 100

        # Also, separate scores:
        hyperactivity_score = (rapid_face_indicator * 50 + rapid_eye_indicator * 50)  # up to 100
        attention_score = blink_indicator * 100  # either 0 or 100
        
        return {
            'hyperactivity_score': hyperactivity_score,
            'attention_score': attention_score,
            'total_adhd_score': total_indicator,
            'blink_rate': blink_rate,
            'rapid_face_proportion': rapid_face_proportion,
            'rapid_eye_proportion': rapid_eye_proportion
        }

    def assess_adhd_indicators(self):
        indicators = {
            'total_face_movements': sum(1 for m in self.face_movement_history if m['is_moving']),
            'rapid_face_movements': sum(1 for m in self.face_movement_history if m['is_rapid']),
            'total_eye_movements': sum(1 for m in self.eye_movement_history if m['is_moving']),
            'rapid_eye_movements': sum(1 for m in self.eye_movement_history if m['is_rapid']),
            'total_blinks': sum(1 for b in self.blink_history if b['is_blinking']),
            'face_movement_variability': None,
            'eye_movement_variability': None,
            'blink_rate': None
        }
        
        try:
            if self.face_movement_history:
                face_vals = [m['movement_magnitude'] for m in self.face_movement_history]
                indicators['face_movement_variability'] = statistics.stdev(face_vals)
            if self.eye_movement_history:
                eye_vals = [m['movement_magnitude'] for m in self.eye_movement_history]
                indicators['eye_movement_variability'] = statistics.stdev(eye_vals)
        except (statistics.StatisticsError, TypeError):
            pass
        
        if self.blink_history:
            indicators['blink_rate'] = sum(1 for b in self.blink_history if b['is_blinking']) / len(self.blink_history)
        
        indicators.update(self._calculate_adhd_score())
        return indicators

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    movement_analyzer = ADHDMovementAnalyzer(tracking_duration=60)
    start_time = time.time()
    
    while time.time() - start_time < movement_analyzer.tracking_duration:
        ret, frame = cap.read()
        if not ret:
            break

        movement_results = movement_analyzer.analyze_movement(frame)
        
        # Display face movement info
        if movement_results.get('face_movement'):
            fm = movement_results['face_movement']
            face_text = f"Face Move: {fm['movement_magnitude']:.2f}"
            face_color = (0, 255, 0)
            if fm['is_rapid']:
                face_text = "Rapid Face Move!"
                face_color = (0, 0, 255)
            cv2.putText(frame, face_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, face_color, 2)
        
        # Display eye movement info
        if movement_results.get('eye_movement'):
            em = movement_results['eye_movement']
            eye_text = f"Eye Move: {em['movement_magnitude']:.2f}"
            eye_color = (255, 0, 0)
            if em['is_rapid']:
                eye_text = "Rapid Eye Move!"
                eye_color = (0, 0, 255)
            cv2.putText(frame, eye_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)
        
        # Display blink info
        if movement_results.get('blink'):
            bm = movement_results['blink']
            blink_text = f"EAR: {bm['ear']:.2f}"
            blink_color = (0, 255, 255)
            if bm['is_blinking']:
                blink_text = "Blinking!"
                blink_color = (0, 0, 255)
            cv2.putText(frame, blink_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, blink_color, 2)
        
        # Display running ADHD scores
        indicators = movement_analyzer._calculate_adhd_score()
        score_text = (f"Hyper: {indicators['hyperactivity_score']:.1f} "
                      f"Attn: {indicators['attention_score']:.1f} "
                      f"Total: {indicators['total_adhd_score']:.1f}")
        cv2.putText(frame, score_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow('ADHD Movement Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final assessment of ADHD indicators
    adhd_indicators = movement_analyzer.assess_adhd_indicators()
    print("\n--- ADHD Movement Analysis ---")
    print(f"Total Face Movements: {adhd_indicators['total_face_movements']}")
    print(f"Rapid Face Movements: {adhd_indicators['rapid_face_movements']}")
    print(f"Total Eye Movements: {adhd_indicators['total_eye_movements']}")
    print(f"Rapid Eye Movements: {adhd_indicators['rapid_eye_movements']}")
    print(f"Total Blinks: {adhd_indicators['total_blinks']}")
    if adhd_indicators['face_movement_variability'] is not None:
        print(f"Face Movement Variability: {adhd_indicators['face_movement_variability']:.2f}")
    if adhd_indicators['eye_movement_variability'] is not None:
        print(f"Eye Movement Variability: {adhd_indicators['eye_movement_variability']:.2f}")
    if adhd_indicators['blink_rate'] is not None:
        print(f"Blink Rate (per min): {adhd_indicators['blink_rate']:.2f}")
    
    print("\n--- ADHD Potential Indicators ---")
    print(f"Hyperactivity Score: {adhd_indicators['hyperactivity_score']:.2f}")
    print(f"Attention Score: {adhd_indicators['attention_score']:.2f}")
    print(f"Total ADHD Score: {adhd_indicators['total_adhd_score']:.2f}")
    
    # Determine if ADHD potential is detected
    if (adhd_indicators['attention_score'] == 100 and 
        adhd_indicators['hyperactivity_score'] >= 50):
        print("\nADHD potential detected based on the movement patterns.")
    else:
        print("\nADHD potential NOT detected based on the movement patterns.")

if __name__ == "__main__":
    main()
