<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 120" width="520" height="120">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0f0c29"/>
      <stop offset="50%" style="stop-color:#302b63"/>
      <stop offset="100%" style="stop-color:#24243e"/>
    </linearGradient>
  </defs>
  <rect width="520" height="120" rx="12" fill="url(#bg)"/>
  <text x="260" y="52" font-family="Arial,sans-serif" font-size="26" font-weight="bold" fill="#ffffff" text-anchor="middle">🧠 NeuroSight</text>
  <text x="260" y="80" font-family="Arial,sans-serif" font-size="13" fill="#a78bfa" text-anchor="middle">ADHD &amp; Dyslexia Detection via Eye Tracking + AI</text>
  <text x="260" y="105" font-family="Arial,sans-serif" font-size="10" fill="#6b7280" text-anchor="middle">Real-time · MediaPipe · Random Forest · Google Gemini</text>
</svg>

---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?style=flat-square&logo=scikit-learn&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Google%20Gemini-1.5--flash-purple?style=flat-square&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-Research%20Only-red?style=flat-square)

</div>

---

> ⚠️ **Disclaimer**: This is a research and educational tool. It is **not** a medical device and **not** intended for clinical diagnosis of ADHD or dyslexia.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [File Reference](#-file-reference)
- [Requirements](#-requirements)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Output Metrics](#-output-metrics)
- [How It Works](#-how-it-works)
- [Use Cases](#-use-cases)
- [Known Limitations](#-known-limitations)

---

## 🔭 Overview

**NeuroSight** is a Python-based multimodal analysis system that combines:

| Module | Technology | Purpose |
|---|---|---|
| Eye Tracking | MediaPipe FaceMesh | Blink, saccade & fixation metrics |
| ADHD Scoring | Random Forest (scikit-learn) | Classify Neurodiverse vs Neurotypical |
| Dyslexia Analysis | Google Gemini 1.5-flash | Handwriting visual pattern analysis |
| Movement Analysis | dlib + OpenCV | Face & eye movement intensity |

---

## ✨ Features

### 🧠 ADHD Eye-Tracking Analysis (`system.py`, `cam.py`, `detect.py`)
- **Real-time webcam** eye tracking at ~30 fps
- **Blink detection** using Eye Aspect Ratio (EAR) with temporal consistency
- **Saccade detection** — rapid eye jumps detected with hysteresis thresholding
- **Fixation duration** measurement between saccadic movements
- **Gaze direction** estimation (Left / Center / Right) from iris landmarks
- **Random Forest classifier** trained on 13 oculomotor features
- **Multi-tab Tkinter GUI** with live video feed and log panel
- **Video file upload** with progress bar and automatic classification

### 📝 Dyslexia Handwriting Analysis (`system.py`, `imge.py`)
- Upload any handwriting image (PNG, JPG, BMP, GIF)
- AI-powered analysis of **7 dyslexia indicators**:
  - Letter reversals (b/d, p/q)
  - Inconsistent letter formation
  - Omissions or insertions
  - Transpositions (letter swaps)
  - Irregular spacing
  - Poor baseline alignment
  - Visual clutter / disorganization
- Outputs a **0–100% likelihood score** with a structured summary

### 📊 Data Visualization (`graph.py`)
- Box plots comparing ADHD / Dyslexia / Control groups
- Scatter plots: Saccade Frequency vs Fixation Duration

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                    system.py (Main App)              │
│                                                     │
│  ┌──────────────────┐   ┌────────────────────────┐  │
│  │  EyeTrackerGUI   │   │  DyslexiaAnalyzerGUI   │  │
│  │  (Tab 1)         │   │  (Tab 2)               │  │
│  └────────┬─────────┘   └──────────┬─────────────┘  │
│           │                        │                │
│           ▼                        ▼                │
│   MediaPipe FaceMesh        Google Gemini API       │
│   ┌──────────────┐         ┌──────────────────┐     │
│   │  EAR / Iris  │         │  Image → Prompt  │     │
│   │  landmarks   │         │  → Analysis text │     │
│   └──────┬───────┘         └──────────────────┘     │
│          ▼                                          │
│   ┌──────────────────────────┐                      │
│   │  Metrics JSON output     │                      │
│   │  → Random Forest         │                      │
│   │  → ADHD / Neurotypical   │                      │
│   └──────────────────────────┘                      │
└─────────────────────────────────────────────────────┘
```

---

## 📁 File Reference

| File | Description |
|---|---|
| `system.py` | **Main entry point** — tabbed GUI combining eye tracking + dyslexia analysis |
| `detect.py` | Standalone eye-tracking GUI with built-in ML classification |
| `cam.py` | Headless (no GUI) eye-tracker — saves metrics to `improved_eye_metrics.json` |
| `eye.py` | Minimal gaze tracker — saves gaze data to `gaze_data.json` |
| `rapid.py` | dlib-based ADHD movement analyzer with ADHD score calculation |
| `nasla.py` | PyQt5-based GUI wrapper for the dlib movement tracker |
| `imge.py` | Standalone dyslexia image analysis GUI |
| `graph.py` | Data visualization (box plots, scatter plots) |
| `print.py` | Utility to inspect MATLAB `.mat` pupil dataset files |
| `dataset.csv` | Training data with 13 oculomotor features + Label column |
| `adhd_model.pkl` | Pre-trained Random Forest model (auto-generated on first run) |
| `improved_eye_metrics.json` | Eye-tracking metrics output (auto-generated) |
| `gaze_data.json` | Per-frame gaze data output (auto-generated) |

---

## 📦 Requirements

### Python Version
- Python **3.9+** recommended

### Dependencies (`requirements.txt`)

```
opencv-python      # Webcam capture, frame processing, drawing
mediapipe          # FaceMesh — 468+10 iris landmarks
numpy              # Numerical computations
pandas             # Dataset loading and manipulation
scikit-learn       # Random Forest classifier
joblib             # Model serialization
Pillow             # Image display in Tkinter
python-dotenv      # Load .env files for API keys
google-generativeai # Google Gemini API for dyslexia analysis
matplotlib         # Data visualization (graph.py)
seaborn            # Statistical plots (graph.py)
```

**Optional** (for `rapid.py` and `nasla.py` only):
```
dlib               # Face detector + 68-point predictor
PyQt5              # Qt5 GUI framework
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Kaelith69/adhd.git
cd adhd
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download face landmark model *(for `rapid.py` / `nasla.py` only)*
Download `shape_predictor_68_face_landmarks.dat` from the [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract it to the project root.

### 5. Set up your Google API key *(for dyslexia analysis)*
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
> Never commit your `.env` file — it is already listed in `.gitignore`.

---

## ⚙️ Configuration

### Eye Tracking Parameters (in `system.py` / `detect.py` / `cam.py`)

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.2` | Eye Aspect Ratio below which a blink is detected |
| `EAR_CONSEC_FRAMES` | `2` | Consecutive frames below threshold to confirm blink |
| `SACCADE_THRESHOLD` | `10 px` | Eye movement magnitude to trigger a saccade |
| `SACCADE_MIN_DURATION` | `0.02 s` | Minimum saccade duration to count |
| `FIXATION_MIN_DURATION` | `0.1 s` | Minimum fixation duration to record |
| `duration` | `30 s` | Auto-stop time for `cam.py` headless mode |

### ML Classifier Thresholds

| Parameter | Default | Description |
|---|---|---|
| `RAPID_SACCADE_THRESHOLD` | `30 /min` | Saccade frequency above which rapid eye movement is flagged |
| `REPETITIVE_BLINK_THRESHOLD` | `15 /min` | Blink rate above which repetitive blinking is flagged |

---

## 🖥 Usage

### Main Application (recommended)
```bash
python system.py
```
This opens a two-tab window:

**Tab 1 — Eye Tracking & Gaze**
1. Click **"Start Eye Tracking"** to begin live webcam tracking
2. A 30-second session records blinks, saccades, fixations, and gaze direction
3. Click **"Stop"** to end and save metrics to `improved_eye_metrics.json`
4. Click **"Run Classification"** to get an ADHD/Neurotypical prediction
5. Or click **"Upload Video"** to process a pre-recorded `.mp4`/`.avi`/`.mov`

**Tab 2 — Dyslexia Analyzer**
1. Click **"Upload Image"** to select a handwriting sample
2. Click **"Direct Visual Analysis for Dyslexia"**
3. View the AI-generated report and likelihood score

---

### Headless Eye Tracker
```bash
python cam.py
```
- Opens webcam, runs for 30 seconds (or press `q` to quit early)
- Saves all metrics to `improved_eye_metrics.json`
- Displays real-time overlay: EAR, blink count, saccade count, gaze direction

---

### Gaze-Only Tracker
```bash
python eye.py
```
- Tracks iris position and gaze direction
- Press `q` to quit; saves all data to `gaze_data.json`

---

### dlib-Based Movement Analyzer
```bash
python rapid.py
```
- Uses dlib 68-point face model
- Tracks face movement, eye movement, blink rate
- Outputs ADHD hyperactivity/attention/total scores after 60 seconds

---

### Data Visualization
```bash
python graph.py
```
- Generates box plots and scatter plots comparing simulated ADHD / Dyslexia / Control groups

---

### Dyslexia Image Analysis (standalone)
```bash
python imge.py
```
- Standalone GUI for handwriting upload and Gemini-powered dyslexia analysis

---

## 📊 Output Metrics

### `improved_eye_metrics.json` — Eye Tracking Output

```json
{
  "TotalCaptureTime_sec": 30.1,
  "ProcessedFrames": 450,
  "DroppedFrames": 2,
  "FrameRate_fps": 14.9,
  "DropRate_percent": 0.44,
  "BlinkMetrics": {
    "BlinkCount": 8,
    "BlinkRate_per_min": 15.9
  },
  "SaccadeMetrics": {
    "SaccadeCount": 22,
    "SaccadeFrequency_per_min": 43.9,
    "AverageSaccadeAmplitude_pixels": 14.2,
    "MaxSaccadeAmplitude_pixels": 31.5,
    "MinSaccadeAmplitude_pixels": 10.1
  },
  "FixationMetrics": {
    "FixationCount": 21,
    "AverageFixationDuration_sec": 0.58,
    "MaxFixationDuration_sec": 1.2,
    "MinFixationDuration_sec": 0.1,
    "FixationFrequency_per_min": 41.9
  },
  "GazeMetrics": {
    "LastGazeDirection": "Center",
    "IrisInfo": { ... }
  },
  "Parameters": { ... }
}
```

### ML Classification Output
The classifier outputs one of:
- `"ADHD"` — Neurodiverse prediction with ≥2/3 additional indicators
- `"Neurotypical"` — No strong ADHD indicators detected

---

## 🔬 How It Works

### Eye Aspect Ratio (EAR) — Blink Detection

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```
Where p1–p6 are 6 eye landmark points. When EAR drops below `EAR_THRESHOLD` for `EAR_CONSEC_FRAMES` consecutive frames, a blink is registered.

### Saccade Detection
Eye center position is smoothed with a 3-frame moving average. A movement exceeding `SACCADE_THRESHOLD` pixels triggers a saccade. Hysteresis (70% of threshold to exit) prevents double-counting.

### Iris-based Gaze Estimation
MediaPipe provides 5 iris landmarks per eye (468–472 left, 473–477 right). The iris center is compared to eye corner landmarks (33 & 133) to determine Left/Center/Right gaze direction.

### ADHD Classification
A Random Forest classifier is trained on 13 oculomotor features from `dataset.csv`. After a tracking session, the live metrics are flattened and fed to the model. If the base prediction is Neurodiverse, two additional rule-based checks (rapid saccade frequency > 30/min, blink rate > 15/min) are evaluated — a score ≥ 2/3 results in an `"ADHD"` label.

---

## 💡 Use Cases

| Scenario | Recommended Script |
|---|---|
| Research study — screen for ADHD indicators | `system.py` (Tab 1) |
| Analyze pre-recorded session video | `system.py` → Upload Video |
| Analyze handwriting for dyslexia | `system.py` (Tab 2) or `imge.py` |
| Quick headless webcam session | `cam.py` |
| Visualize group differences | `graph.py` |
| Physical movement-based ADHD assessment | `rapid.py` |

---

## ⚠️ Known Limitations

- **Not a medical diagnostic tool** — results should not be used for clinical decisions
- Iris landmark detection requires good, even lighting and a front-facing camera
- Classification accuracy depends on the quality and representativeness of `dataset.csv`
- The `SaccadeLatency`, `SaccadeAmplitudeVariability`, and several other ML features default to `0` when computed from live tracking — a richer sensor setup would improve accuracy
- `rapid.py` and `nasla.py` require `dlib` and `shape_predictor_68_face_landmarks.dat` which are not installed by default
- Dyslexia analysis requires a valid `GOOGLE_API_KEY` in your `.env` file

---

## 📄 License

This project is for **research and educational purposes only**. Not licensed for commercial or clinical use. See individual file headers for attribution.
