# ADHD and Dyslexia Analysis System

A comprehensive Python-based system for analyzing ADHD indicators through eye tracking and dyslexia through handwriting analysis.

## Features

### ADHD Analysis
- Real-time eye movement tracking
- Facial landmark detection
- Metrics tracked:
  - Blink rate and patterns
  - Saccadic eye movements
  - Gaze direction and fixation
  - Movement intensity
  - Face movement patterns

### Dyslexia Analysis
- Handwriting analysis using Google's Gemini AI
- Visual indicators evaluated:
  - Letter reversals (b/d, p/q)
  - Inconsistent letter formation
  - Spacing irregularities
  - Baseline alignment
  - Visual organization

## Requirements

```sh
pip install -r requirements.txt
```

Key dependencies:
- OpenCV
- MediaPipe
- dlib
- NumPy
- scikit-learn
- PyQt5/Tkinter (GUI)
- Google Generative AI
- Pillow

## Setup

1. Install dependencies
2. Download the face landmark predictor:
   - `shape_predictor_68_face_landmarks.dat`
3. Set up Google API key:
   ```python
   GOOGLE_API_KEY = "your_api_key_here"
   ```

## Usage

### ADHD Analysis
```sh
python system.py
```
- Select "Eye Tracking & Gaze" tab
- Follow on-screen instructions for eye movement analysis

### Dyslexia Analysis
```sh
python system.py
```
- Select "Dyslexia Analyzer" tab
- Upload handwriting sample
- Click "Analyze" for AI-powered assessment

## Key Components

- `system.py`: Main application with GUI
- `rapid.py`: ADHD movement analysis
- `cam.py`: Eye tracking implementation
- `imge.py`: Dyslexia analysis interface
- `detect.py`: ML-based detection system
- `eye.py`: Gaze tracking utilities

## Output

### ADHD Metrics
- JSON output with comprehensive metrics
- Real-time visualization
- Movement pattern analysis
- Statistical indicators

### Dyslexia Assessment
- AI-generated analysis report
- Probability score
- Visual indicator breakdown

## Notes

- This is a research/educational tool
- Not intended for medical diagnosis
- Requires proper lighting and camera setup
- Best used in controlled environments

## License

[Add your license information here]