import sys
import cv2
import numpy as np
import dlib
import time
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class ADHDTracker(QWidget):
    def __init__(self):
        super().__init__()
        
        self.initUI()
        self.capture = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.movement_intensity = []
        self.blinks = 0
        self.last_blink_time = time.time()

    def initUI(self):
        self.setWindowTitle("ADHD Gaze Tracker")
        self.setGeometry(100, 100, 800, 600)
        
        self.video_label = QLabel(self)
        self.start_btn = QPushButton("Start", self)
        self.stop_btn = QPushButton("Stop", self)
        self.save_btn = QPushButton("Save Results", self)
        
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.save_btn.clicked.connect(self.save_results)
        
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
    
    def start_tracking(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.start(30)
    
    def stop_tracking(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
    
    def save_results(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt)")
        if filename:
            with open(filename, "w") as file:
                file.write(f"Total Blinks: {self.blinks}\n")
                file.write(f"Movement Intensity: {self.movement_intensity}\n")
            print("Results saved.")
    
    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            for face in faces:
                landmarks = self.predictor(gray, face)
                eye_top = landmarks.part(37).y
                eye_bottom = landmarks.part(41).y
                eye_height = abs(eye_bottom - eye_top)
                
                if eye_height < 5:  # Blink detection threshold
                    if time.time() - self.last_blink_time > 0.2:
                        self.blinks += 1
                        self.last_blink_time = time.time()
                
                movement = np.random.randint(1, 10)  # Placeholder for actual movement calculation
                self.movement_intensity.append(movement)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        self.stop_tracking()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ADHDTracker()
    window.show()
    sys.exit(app.exec_())