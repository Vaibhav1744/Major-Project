import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import wavio
import os
from tensorflow.keras.models import load_model

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "Emotions_Model.h5"
model = load_model(MODEL_PATH)

# -----------------------------
# Define class mapping (14 classes)
# -----------------------------
emotion_dict = {
    0: "male_neutral",
    1: "female_neutral",
    2: "male_sad",
    3: "female_sad",
    4: "male_happy",
    5: "female_happy",
    6: "male_angry",
    7: "female_angry",
    8: "male_disgust",
    9: "female_disgust",
    10: "male_fear",
    11: "female_fear",
    12: "male_surprise",
    13: "female_surprise"
}

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(audio_file, max_pad_len=216):
    y, sr = librosa.load(audio_file, sr=None)

    # Use a single MFCC coefficient (1D feature sequence)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)

    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0), (0,pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = mfcc.T   # shape → (216, 1)
    return np.expand_dims(mfcc, axis=0)   # shape → (1, 216, 1)

# -----------------------------
# Predict emotion
# -----------------------------
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    preds = model.predict(features)
    predicted_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    emotion = emotion_dict.get(predicted_index, f"Unknown ({predicted_index})")
    return emotion, confidence

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎤 Real-time Voice Emotion Recognition")

# Record voice
duration = st.slider("Recording Duration (seconds)", 2, 10, 4)
fs = 16000
record_btn = st.button("🎙️ Record Voice")

if record_btn:
    st.info("Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    wav_output = "recorded_voice.wav"
    wavio.write(wav_output, recording, fs, sampwidth=2)
    st.success("Recording complete! ✅")

    # Predict
    emotion, confidence = predict_emotion(wav_output)
    st.success(f"🎭 Detected Emotion: **{emotion}** (Confidence: {confidence:.2f})")

# Upload audio file
uploaded_file = st.file_uploader("Or upload a WAV file", type=["wav"])
if uploaded_file is not None:
    temp_path = "uploaded_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    emotion, confidence = predict_emotion(temp_path)
    st.success(f"🎭 Detected Emotion: **{emotion}** (Confidence: {confidence:.2f})")
