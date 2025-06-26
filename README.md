
# 🖐️ Real-Time Gesture Recognition with Machine Learning

This project is a real-time gesture recognition system using **MediaPipe**, **OpenCV**, and **Scikit-learn**, wrapped in a browser-friendly **Streamlit** app. It allows you to collect your own hand gesture data, train a custom ML model, and use it to make live predictions from your webcam.

---

## Features

- Real-time hand tracking using **MediaPipe**
- Custom gesture data collection (`.npy`)
- ML model training using **RandomForestClassifier**
- Live gesture prediction using webcam input
- Streamlit web app UI for portfolio/demo purposes

---

## Project Structure

```
gesture_app/
├── app.py                  # Streamlit app (Step 5)
├── .gitignore              # Ignore .npy, .pkl, venv files
├── utils/
│   ├── hand_detector.py    # Live hand tracking test (Step 1)
│   ├── data_collector.py   # Save gesture data as .npy (Step 2)
│   ├── train_model.py      # Train classifier (Step 3)
│   ├── predict_live.py     # Predict gesture in real time (Step 4)
│   ├── inspect_data.py     # Inspect .npy files
│   └── inspect_model.py    # Load and inspect .pkl model
├── data/                   # Contains gesture data like 0.npy, 1.npy
├── models/                 # Saved trained model (.pkl)
└── requirements.txt        # All dependencies
```

---

##  Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/gesture-recognition-app.git
cd gesture-recognition-app
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, run:

```bash
pip install opencv-python mediapipe streamlit numpy pandas scikit-learn joblib
```

---

## Step-by-Step Usage

### Step 1: Hand Tracking Test

```bash
python utils/hand_detector.py
```
This opens your webcam and visualizes hand landmarks. Press `q` to quit.

---

### Step 2: Collect Gesture Data

```bash
python utils/data_collector.py
```

- Input a gesture label (e.g., `0` for fist, `1` for peace)
- Hold the gesture steady to collect samples (default: 100)
- Data is saved to `data/0.npy`, `data/1.npy`, etc.

---

### Step 3: Train the Gesture Classifier

```bash
python utils/train_model.py
```

- Loads all `.npy` files in `data/`
- Trains a `RandomForestClassifier`
- Saves the model to `models/gesture_model.pkl`
- Prints training accuracy + evaluation report

---

### Optional: Inspect Model or Data

```bash
python utils/inspect_data.py       # View shape & sample from .npy
python utils/inspect_model.py      # View model structure and parameters
```

---

### Step 4: Live Prediction (Command Line)

```bash
python utils/predict_live.py
```

- Uses webcam and predicts trained gestures in real-time
- Shows prediction and confidence score on screen

---

### Step 5: Run the Streamlit Web App

```bash
streamlit run app.py
```

- Opens [localhost:8501](http://localhost:8501) in your browser
- Toggle checkbox to start webcam
- Display your hand to see predictions and live confidence

---

## Customization Tips

- Add more gestures by collecting `2.npy`, `3.npy`, etc.
- Retrain the model with more varied examples for better confidence
- Replace `RandomForestClassifier` with another ML algorithm if desired
- Enhance Streamlit UI with gesture-triggered actions

---

## License

This project is open-source for learning and demonstration purposes. Use it freely for personal use.

---

## Built With

- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) – Real-time hand landmark detection
- [OpenCV](https://opencv.org/) – Webcam access and image processing
- [Scikit-learn](https://scikit-learn.org/) – Machine learning backend (Random Forest Classifier)
- [Streamlit](https://streamlit.io/) – Web-based UI framework
- [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/) – Data manipulation
- [Joblib](https://joblib.readthedocs.io/) – Saving/loading ML models


---

Built by Pavlo Sernetskyi
