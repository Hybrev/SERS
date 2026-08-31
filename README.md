# SERS — Speech Emotion Recognition System (Thesis App)

A Streamlit application that detects emotions from speech recordings using deep-learning models trained on the RAVDESS dataset.

## Features

- Record or upload WAV audio
- Classify emotions such as happy, sad, angry, fearful, and neutral
- Compare Xception, ResNet-50, and LSTM architectures
- Explore model performance, experiments, and visualizations

## Tech Stack

Python 3.10 · TensorFlow/Keras · Streamlit · Librosa · Scikit-learn

## Run Locally

```bash
git clone https://github.com/Hybrev/SERS.git
cd SERS
git lfs pull
pip install -r requirements.txt
streamlit run main.py
```

> This is an experimental research project. Prediction accuracy may vary depending on recording quality, background noise, and speaker characteristics.
