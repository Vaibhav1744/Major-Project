# app.py
"""
Emotion Music App — updated full version
Features:
- Login / Register / Onboarding (simple fallback auth included)
- Face emotion detection (MTCNN fallback to Haar cascade)
- Live webcam preview inside Streamlit while scanning
- Voice recording/upload + waveform visualization
- Face+Voice fusion with adjustable weights
- Mood-improvement prompt and YouTube recommendations (optional API)
- Sidebar Reset / status + preferences
Requirements (example):
pip install streamlit opencv-python-headless tensorflow librosa sounddevice wavio python-dotenv google-api-python-client matplotlib mtcnn
"""
import os
import time
import random
import logging
from collections import Counter
import io

import numpy as np
import streamlit as st
import cv2
import librosa
import sounddevice as sd
import wavio
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Streamlit config — must be before other Streamlit calls
st.set_page_config(page_title="Emotion Music App", page_icon="🎵", layout="wide")

# ML libs
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Optional MTCNN
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except Exception:
    MTCNN_AVAILABLE = False

# Optional external modules (auth/db)
try:
    from auth import register_user, login_user
    from database import users_collection
    AUTH_AVAILABLE = True
except Exception:
    AUTH_AVAILABLE = False

# YouTube API optional
try:
    from googleapiclient.discovery import build
    GOOGLEAPI_AVAILABLE = True
except Exception:
    GOOGLEAPI_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotion-music")

# Load .env
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
youtube = None
if YOUTUBE_API_KEY and GOOGLEAPI_AVAILABLE:
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        logger.warning("YouTube API init failed: %s", e)
        youtube = None

# -------------------------
# Fallback auth
# -------------------------
if not AUTH_AVAILABLE:
    _USERS = {}
    def register_user(email, username, password):
        if username in _USERS:
            return False, "Username already exists"
        _USERS[username] = {"username": username, "email": email, "password": password, "preferences": {}}
        return True, "Registered successfully"
    def login_user(username_or_email, password):
        for u in _USERS.values():
            if u["username"] == username_or_email or u["email"] == username_or_email:
                if u["password"] == password:
                    return True, "Login success", u
                return False, "Invalid password", None
        return False, "User not found", None
    class _DummyCol:
        def update_one(self, q, s):
            return
    users_collection = _DummyCol()

# -------------------------
# Emotion dictionaries
# -------------------------
FACE_EMOTION_DICT = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprised"
}

VOICE_EMOTION_DICT_14 = {
    0: "male_neutral", 1: "female_neutral", 2: "male_sad", 3: "female_sad",
    4: "male_happy", 5: "female_happy", 6: "male_angry", 7: "female_angry",
    8: "male_disgust", 9: "female_disgust", 10: "male_fear", 11: "female_fear",
    12: "male_surprise", 13: "female_surprise"
}

VOICE_TO_COMMON = {
    "male_neutral":"Neutral","female_neutral":"Neutral",
    "male_sad":"Sad","female_sad":"Sad",
    "male_happy":"Happy","female_happy":"Happy",
    "male_angry":"Angry","female_angry":"Angry",
    "male_disgust":"Disgusted","female_disgust":"Disgusted",
    "male_fear":"Fearful","female_fear":"Fearful",
    "male_surprise":"Surprised","female_surprise":"Surprised"
}

COMMON_EMOTIONS = ["Angry","Disgusted","Fearful","Happy","Neutral","Sad","Surprised"]

MOOD_IMPROVEMENT_MAP = {
    "Angry":"Calm","Disgusted":"Happy","Fearful":"Devotional",
    "Happy":"Happy","Neutral":None,"Sad":"Happy","Surprised":"Happy"
}

# -------------------------
# Session defaults
# -------------------------
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user' not in st.session_state: st.session_state.user = None
if 'page' not in st.session_state: st.session_state.page = 'login'
if 'emotion_list' not in st.session_state: st.session_state.emotion_list = []
if 'voice_label' not in st.session_state: st.session_state.voice_label = None
if 'voice_common' not in st.session_state: st.session_state.voice_common = None
if 'voice_conf' not in st.session_state: st.session_state.voice_conf = None
if 'voice_debug' not in st.session_state: st.session_state.voice_debug = None
if 'current_songs' not in st.session_state: st.session_state.current_songs = []
if 'playlist_url' not in st.session_state: st.session_state.playlist_url = None
if 'mood_choice' not in st.session_state: st.session_state.mood_choice = None
if 'preferences' not in st.session_state:
    st.session_state.preferences = {"song_type":"Pop","language":"English"}

# -------------------------
# Model loaders (cached)
# -------------------------
@st.cache_resource
def load_face_emotion_model(path="model.h5"):
    if not os.path.exists(path):
        logger.warning("Face model file not found: %s", path)
        return None
    # build architecture identical to training
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    try:
        model.load_weights(path)
        logger.info("Face model weights loaded")
        return model
    except Exception as e:
        logger.error("Failed to load face model weights: %s", e)
        return None

@st.cache_resource
def load_voice_model(path="Emotions_Model.h5"):
    if not os.path.exists(path):
        logger.warning("Voice model file not found: %s", path)
        return None
    try:
        m = load_model(path)
        logger.info("Voice model loaded")
        return m
    except Exception as e:
        logger.error("Failed to load voice model: %s", e)
        return None

# -------------------------
# Voice feature pipeline (improved)
# -------------------------
def pre_emphasize(y, coef=0.97):
    if y.size == 0:
        return y
    return np.append(y[0], y[1:] - coef * y[:-1])

def audio_denoise_simple(y):
    if y.size == 0:
        return y
    mx = np.max(np.abs(y))
    if mx > 0:
        y = y / mx
    y[np.abs(y) < 1e-4] = 0.0
    return y

def compute_first_pc(feature_matrix):
    F, T = feature_matrix.shape
    if F == 0 or T == 0:
        return np.zeros(T, dtype=np.float32)
    try:
        U, S, Vt = np.linalg.svd(feature_matrix - np.mean(feature_matrix, axis=1, keepdims=True), full_matrices=False)
        pc = U[:,0]
        projected = pc.T @ (feature_matrix - np.mean(feature_matrix, axis=1, keepdims=True))
        return projected
    except Exception:
        return np.mean(feature_matrix, axis=0)

def extract_voice_features_improved(path, sr_target=16000, n_mfcc=40, target_len=216, pre_emph=True, denoise=True):
    """Return (features, debug) where debug includes original waveform for plotting."""
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    if y.size == 0:
        y = np.zeros(int(sr_target*2.0), dtype=np.float32)
    if pre_emph:
        y = pre_emphasize(y, coef=0.97)
    if denoise:
        y = audio_denoise_simple(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack([mfcc, d1, d2])
    stacked = np.nan_to_num(stacked)
    seq = compute_first_pc(stacked)
    if seq.shape[0] < target_len:
        seq = np.pad(seq, (0, target_len - seq.shape[0]), mode='constant')
    else:
        seq = seq[:target_len]
    if np.std(seq) > 1e-6:
        seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-9)
    x = seq.reshape(1, target_len, 1).astype(np.float32)
    debug = {"sr": sr, "orig_len": len(y), "stacked_shape": stacked.shape, "final_shape": x.shape, "waveform": y}
    return x, debug

def voice_predict_smoothed(vmodel, audio_path, cfg):
    feats, dbg = extract_voice_features_improved(
        audio_path,
        sr_target=cfg.get("sr",16000),
        n_mfcc=cfg.get("n_mfcc",40),
        target_len=cfg.get("target_len",216),
        pre_emph=cfg.get("pre_emph", True),
        denoise=cfg.get("denoise", True)
    )
    preds = vmodel.predict(feats, verbose=0)  # shape (1,14)
    avg = np.mean(preds, axis=0)
    idx = int(np.argmax(avg))
    conf = float(np.max(avg))
    return idx, conf, dbg, avg

# -------------------------
# Fusion helpers
# -------------------------
def get_face_scores(emotion_list):
    counts = Counter(emotion_list)
    total = sum(counts.values()) or 1
    return {e: counts.get(e, 0)/total for e in COMMON_EMOTIONS}

def get_voice_scores(voice_common, confidence):
    scores = {e: 0.0 for e in COMMON_EMOTIONS}
    if voice_common in scores:
        scores[voice_common] = float(confidence)
    return scores

def fuse_emotions(face_scores, voice_scores, w_face=0.6, w_voice=0.4):
    fused = {}
    for e in COMMON_EMOTIONS:
        fused[e] = w_face * face_scores.get(e, 0.0) + w_voice * voice_scores.get(e, 0.0)
    total = sum(fused.values()) or 1.0
    fused_norm = {k: v/total for k,v in fused.items()}
    best = max(fused_norm, key=fused_norm.get)
    return best, fused_norm

# -------------------------
# Video and audio helpers
# -------------------------
def record_audio_to_file(duration=3, fs=16000, out="voice_input.wav"):
    try:
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        wavio.write(out, rec, fs, sampwidth=2)
        return out
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

# -------------------------
# YouTube helpers
# -------------------------
def get_youtube_recommendations(emotion, song_type, language, limit=10, offset=0):
    if youtube is None:
        return []
    query = f"{emotion.lower()} {song_type.lower()} {language.lower()} music"
    try:
        req = youtube.search().list(q=query, part="snippet", type="video", maxResults=50)
        res = req.execute()
        items = res.get('items', [])
        songs = [{
            'name': it['snippet']['title'],
            'artist': it['snippet']['channelTitle'],
            'link': f"https://www.youtube.com/watch?v={it['id']['videoId']}",
            'video_id': it['id']['videoId']
        } for it in items]
        if offset:
            songs = songs[offset:offset+limit]
        else:
            songs = songs[:limit]
        random.shuffle(songs)
        return songs
    except Exception as e:
        logger.error("YT fetch error: %s", e)
        return []

def generate_playlist_url(songs):
    if not songs: return None
    video_ids = ','.join(s['video_id'] for s in songs)
    return f"https://www.youtube.com/embed/{songs[0]['video_id']}?playlist={video_ids}&autoplay=1&loop=1"

# -------------------------
# UI: Login / Register / Onboarding
# -------------------------
def show_login_page():
    st.markdown("<h2 style='text-align:center;'>Login to Emotion Music</h2>", unsafe_allow_html=True)
    username_or_email = st.text_input("Username or Email")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if not username_or_email or not password:
                st.error("Please enter both username/email and password")
                return
            success, message, user_data = login_user(username_or_email, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user = user_data
                # load preferences from user if present
                st.session_state.preferences = user_data.get('preferences', st.session_state.preferences)
                st.session_state.page = 'main' if user_data.get('preferences') else 'onboarding'
                st.success("Login successful")
                st.rerun()
            else:
                st.error(message)
    with col2:
        if st.button("Sign Up"):
            st.session_state.page = 'register'
            st.experimental_rerun()

def show_register_page():
    st.markdown("<h2 style='text-align:center;'>Create an Account</h2>", unsafe_allow_html=True)
    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Register"):
            if not email or not username or not password or not confirm_password:
                st.error("Please fill all fields"); return
            if password != confirm_password:
                st.error("Passwords do not match"); return
            success, message = register_user(email, username, password)
            if success:
                st.success(message)
                st.session_state.logged_in = True
                st.session_state.user = {"username": username, "email": email, "preferences": {}}
                st.session_state.preferences = {"song_type":"Pop","language":"English"}
                st.session_state.page = 'onboarding'
                st.experimental_rerun()
            else:
                st.error(message)
    with col2:
        if st.button("Back to Login"):
            st.session_state.page = 'login'
            st.experimental_rerun()

def show_onboarding_page():
    st.markdown("<h2 style='text-align:center;'>Set Your Music Preferences</h2>", unsafe_allow_html=True)
    user_display = st.session_state.user['username'] if st.session_state.user else "User"
    st.write(f"Welcome, {user_display}! Let's set your music preferences.")
    song_type = st.selectbox("What type of songs do you prefer?", ["Devotional","Romantic","Indie","Pop","Rock"], index=["Devotional","Romantic","Indie","Pop","Rock"].index(st.session_state.preferences.get("song_type","Pop")))
    language = st.selectbox("What language do you prefer for music?", ["Hindi","English","Others"], index=["Hindi","English","Others"].index(st.session_state.preferences.get("language","English")))
    if st.button("Save Preferences"):
        preferences = {"song_type": song_type, "language": language}
        try:
            users_collection.update_one({"username": st.session_state.user['username']}, {"$set": {"preferences": preferences}})
        except Exception as e:
            logger.debug("DB update failed: %s", e)
        st.session_state.user['preferences'] = preferences
        st.session_state.preferences = preferences
        st.session_state.page = 'main'
        st.success("Preferences saved successfully!")
        st.experimental_rerun()

# -------------------------
# Main integrated app (UI improved + features requested)
# -------------------------
def show_main_app():
    # load models on demand (cached)
    face_model = load_face_emotion_model()
    voice_model = load_voice_model()

    # Top bar + header area
    st.markdown("<div style='display:flex; align-items:center; justify-content:space-between;'>"
                "<h1>🎵 Emotion Music — Face + Voice</h1>"
                f"<div style='text-align:right;'>User: <b>{st.session_state.user['username'] if st.session_state.user else 'Guest'}</b></div>"
                "</div>", unsafe_allow_html=True)
    st.write("---")

    # Sidebar (top) controls + reset
    st.sidebar.title("Controls & Status")
    if st.sidebar.button("🔄 Clear All / Reset"):
        st.session_state.emotion_list = []
        st.session_state.voice_label = None
        st.session_state.voice_common = None
        st.session_state.voice_conf = None
        st.session_state.voice_debug = None
        st.session_state.current_songs = []
        st.session_state.playlist_url = None
        st.session_state.mood_choice = None
        st.success("Reset complete")
        st.experimental_rerun()

    # Show current mood + last detected emotion in sidebar
    st.sidebar.subheader("Current Status")
    last_emotion = st.session_state.emotion_list[-1] if st.session_state.emotion_list else "None"
    st.sidebar.write(f"**Last Detected Emotion:** {last_emotion}")
    st.sidebar.write(f"**Current Mood Choice:** {st.session_state.mood_choice if st.session_state.mood_choice else 'None'}")

    # Advanced voice settings hidden in sidebar expander
    with st.sidebar.expander("🎙 Advanced Voice Input Settings (click to open)", expanded=False):
        n_mfcc = st.slider("MFCC (n_mfcc)", 13, 60, 40)
        pre_emph = st.checkbox("Apply pre-emphasis (voice)", value=True)
        denoise_opt = st.checkbox("Apply simple denoise", value=True)
        target_len = st.slider("Voice timesteps (target_len)", 120, 300, 216)
        w_face = st.slider("Face weight (fusion)", 0.0, 1.0, 0.6)
        w_voice = st.slider("Voice weight (fusion)", 0.0, 1.0, 0.4)

    # Preferences & quick actions row
    prefs_col, right_col = st.columns([3,1])
    with prefs_col:
        st.markdown("### Preferences")
        st.write("Edit quick preferences (saved locally in session).")
        song_type = st.selectbox("Preferred song type", ["Devotional","Romantic","Indie","Pop","Rock"], index=["Devotional","Romantic","Indie","Pop","Rock"].index(st.session_state.preferences.get("song_type","Pop")))
        language = st.selectbox("Preferred language", ["Hindi","English","Others"], index=["Hindi","English","Others"].index(st.session_state.preferences.get("language","English")))
        st.session_state.preferences["song_type"] = song_type
        st.session_state.preferences["language"] = language
    with right_col:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.page = 'login'
            # clear state related to detection
            st.session_state.emotion_list = []
            st.session_state.voice_label = None
            st.session_state.voice_common = None
            st.session_state.voice_conf = None
            st.experimental_rerun()

    st.write("---")

    # Detection mode and preview area
    settings_col, info_col = st.columns([1,2])
    with settings_col:
        st.header("Settings")
        detection_mode = st.radio("Detection mode", ("Auto (Face + Voice)", "Face only", "Voice only", "Manual"))
        show_snapshot = st.checkbox("Show final snapshot after face scan", value=False)
        # If advanced settings expander was not opened, still ensure variables exist with defaults
        if 'n_mfcc' not in locals():
            n_mfcc = 40
        if 'pre_emph' not in locals():
            pre_emph = True
        if 'denoise_opt' not in locals():
            denoise_opt = True
        if 'target_len' not in locals():
            target_len = 216
        if 'w_face' not in locals():
            w_face = 0.6
        if 'w_voice' not in locals():
            w_voice = 0.4

    with info_col:
        st.header("Status")
        st.write("Last face emotions collected:", st.session_state.emotion_list[-10:] if st.session_state.emotion_list else "None yet")
        st.write("Last voice label:", st.session_state.voice_label if st.session_state.voice_label else "None yet")
        st.write("Last voice common emotion:", st.session_state.voice_common if st.session_state.voice_common else "None yet")

    st.write("---")

    # Manual selection
    st.subheader("Manual emotion (optional)")
    manual_emotion = st.selectbox("Select emotion manually (overrides scans if chosen):", ["None"] + COMMON_EMOTIONS)
    if manual_emotion != "None" and st.button("Use Manual Emotion"):
        st.session_state.emotion_list = [manual_emotion]
        st.success(f"Manual emotion set to: {manual_emotion}")

    # Face scanning area — uses live capture loop to append multiple frames (no external window)
    st.subheader("Step 1 — Face Emotion Scan")
    face_col, voice_col = st.columns(2)
    final_snapshot = None
    with face_col:
        st.write("Press 'Scan Face Emotion' to capture ~30 frames from webcam and run predictions on detected faces. A live preview will show during scanning.")
        if st.button("Scan Face Emotion"):
            if face_model is None:
                st.error("Face model (model.h5) not found or failed to load. Place it in the project folder.")
            else:
                # select detector
                detector = None
                if MTCNN_AVAILABLE:
                    detector = MTCNN()
                else:
                    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    detector = cv2.CascadeClassifier(haar_path)
                cap = cv2.VideoCapture(0)
                collected_emotions = []
                preview_slot = st.empty()
                with st.spinner("Scanning face — keep your face visible to the camera..."):
                    frames = 0
                    while frames < 30:
                        ret, frame = cap.read()
                        if not ret:
                            frames += 1
                            time.sleep(0.03)
                            continue
                        frames += 1
                        display_frame = frame.copy()
                        try:
                            if MTCNN_AVAILABLE:
                                faces = detector.detect_faces(frame)
                                boxes = [f['box'] for f in faces] if faces else []
                            else:
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
                                boxes = [ (x,y,w,h) for (x,y,w,h) in rects ]
                            for box in boxes:
                                x,y,w,h = box
                                x,y = max(0,int(x)), max(0,int(y))
                                roi = frame[y:y+h, x:x+w]
                                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                resized = cv2.resize(gray_roi, (48,48))
                                arr = resized.astype('float32')/255.0
                                arr = arr.reshape(1,48,48,1)
                                preds = face_model.predict(arr, verbose=0)
                                lab = FACE_EMOTION_DICT[int(np.argmax(preds))]
                                collected_emotions.append(lab)
                                # annotate display_frame
                                cv2.rectangle(display_frame, (x,y), (x+w,y+h), (0,255,0), 2)
                                cv2.putText(display_frame, lab, (x, max(y-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
                                final_snapshot = frame.copy()
                        except Exception as e:
                            logger.debug("Face detect/predict error: %s", e)
                        # show a live frame in the UI
                        preview_slot.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                        time.sleep(0.03)
                cap.release()
                if collected_emotions:
                    st.session_state.emotion_list = st.session_state.emotion_list + collected_emotions
                    st.success(f"Face scan complete — {len(collected_emotions)} frames detected.")
                    if show_snapshot and final_snapshot is not None:
                        st.image(cv2.cvtColor(final_snapshot, cv2.COLOR_BGR2RGB), caption="Final snapshot from scan", use_container_width=True)
                else:
                    st.warning("No face/emotion detected—check lighting and camera position.")

    # Voice scanning area
    with voice_col:
        st.subheader("Step 2 — Voice Emotion Scan")
        st.write("Record or upload a WAV file. After detection, a waveform will be shown.")
        rec_dur = st.number_input("Recording duration (seconds)", min_value=2, max_value=8, value=4, step=1)
        if st.button("🎤 Record Voice Now"):
            tmp = record_audio_to_file(duration=int(rec_dur), fs=16000, out="voice_input.wav")
            if tmp:
                vmodel = load_voice_model()
                if vmodel is None:
                    st.error("Voice model (Emotions_Model.h5) not found or failed to load.")
                else:
                    cfg = {"n_mfcc": n_mfcc, "pre_emph": pre_emph, "denoise": denoise_opt, "target_len": target_len}
                    try:
                        idx, conf, dbg, avg = voice_predict_smoothed(vmodel, tmp, cfg)
                        raw_label = VOICE_EMOTION_DICT_14.get(idx, f"Unknown({idx})")
                        common = VOICE_TO_COMMON.get(raw_label, None)
                        st.session_state.voice_label = raw_label
                        st.session_state.voice_common = common
                        st.session_state.voice_conf = conf
                        st.session_state.voice_debug = dbg
                        # append voice common label to emotion_list as common label if available
                        if common:
                            st.session_state.emotion_list = st.session_state.emotion_list + [common]
                        st.audio(tmp, format="audio/wav")
                        st.success(f"Voice -> {raw_label} ({common}) conf={conf:.3f}")
                        # Plot waveform
                        waveform = dbg.get("waveform", None)
                        if waveform is not None:
                            fig, ax = plt.subplots(figsize=(6,2))
                            ax.plot(np.linspace(0, len(waveform)/dbg['sr'], num=len(waveform)), waveform)
                            ax.set_xlabel("Seconds")
                            ax.set_ylabel("Amplitude")
                            ax.set_title("Recorded Waveform")
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Voice prediction failed: {e}")

        uploaded = st.file_uploader("Or upload a WAV file", type=["wav"])
        if uploaded is not None:
            tmpf = "uploaded_voice.wav"
            with open(tmpf, "wb") as f:
                f.write(uploaded.read())
            st.audio(tmpf, format="audio/wav")
            vmodel = load_voice_model()
            if vmodel is None:
                st.error("Voice model not available.")
            else:
                cfg = {"n_mfcc": n_mfcc, "pre_emph": pre_emph, "denoise": denoise_opt, "target_len": target_len}
                try:
                    idx, conf, dbg, avg = voice_predict_smoothed(vmodel, tmpf, cfg)
                    raw_label = VOICE_EMOTION_DICT_14.get(idx, f"Unknown({idx})")
                    common = VOICE_TO_COMMON.get(raw_label, None)
                    st.session_state.voice_label = raw_label
                    st.session_state.voice_common = common
                    st.session_state.voice_conf = conf
                    st.session_state.voice_debug = dbg
                    if common:
                        st.session_state.emotion_list = st.session_state.emotion_list + [common]
                    st.success(f"Voice -> {raw_label} ({common}) conf={conf:.3f}")
                    # waveform
                    waveform = dbg.get("waveform", None)
                    if waveform is not None:
                        fig, ax = plt.subplots(figsize=(6,2))
                        ax.plot(np.linspace(0, len(waveform)/dbg['sr'], num=len(waveform)), waveform)
                        ax.set_xlabel("Seconds")
                        ax.set_ylabel("Amplitude")
                        ax.set_title("Uploaded Waveform")
                        st.pyplot(fig)
                    if st.checkbox("Show voice debug info"):
                        st.json(dbg)
                except Exception as e:
                    st.error(f"Voice prediction failed: {e}")

    # ---- Step 3: Fusion / Manual / Mode handling
    st.subheader("Step 3 — Final emotion (fusion / manual)")
    driving_emotion = None

    # Choose final based on detection_mode
    if detection_mode == "Manual":
        if manual_emotion != "None":
            driving_emotion = manual_emotion
            st.info(f"Manual emotion chosen: {driving_emotion}")
        else:
            st.info("Manual mode selected: choose an emotion from the dropdown and click 'Use Manual Emotion' above.")
    else:
        # For Auto mode we require both at least one face detection AND voice detection before suggestions
        face_scores = get_face_scores([e for e in st.session_state.emotion_list if e in COMMON_EMOTIONS]) if st.session_state.emotion_list else {e:0.0 for e in COMMON_EMOTIONS}
        voice_scores = get_voice_scores(st.session_state.voice_common, st.session_state.voice_conf) if st.session_state.voice_common else {e:0.0 for e in COMMON_EMOTIONS}

        if detection_mode == "Face only":
            driving_emotion = get_top_from_scores(face_scores)
        elif detection_mode == "Voice only":
            driving_emotion = st.session_state.voice_common
        else:  # Auto (Face + Voice)
            # require both modalities present
            if (not st.session_state.emotion_list) or (not st.session_state.voice_common):
                st.info("Auto mode requires both face and voice scans. Run both scans to produce recommendations.")
            else:
                best, fused = fuse_emotions(face_scores, voice_scores, w_face=float(w_face), w_voice=float(w_voice))
                driving_emotion = best
                with st.expander("Fusion details (face / voice / fused)"):
                    st.write("Face scores:", face_scores)
                    st.write("Voice scores:", voice_scores)
                    st.json(fused)

    if driving_emotion:
        st.success(f"Detected Emotion for music: **{driving_emotion}**")
        # update last-detected in session state
        if st.session_state.emotion_list == [] or (st.session_state.emotion_list and st.session_state.emotion_list[-1] != driving_emotion):
            st.session_state.emotion_list = st.session_state.emotion_list + [driving_emotion]
    else:
        # keep informative message if nothing yet
        pass

    # Mood-improvement prompt and recommendation
    adjusted_song_type = None
    if driving_emotion:
        if st.session_state.mood_choice is None:
            adjusted_song_type = prompt_for_mood_improvement_ui(driving_emotion)
            if adjusted_song_type is None:
                st.info("Choose your mood options to continue.")
        else:
            adjusted_song_type = st.session_state.mood_choice

    st.markdown("### Recommended Songs")
    song_type_pref = st.session_state.preferences.get("song_type", "Pop")
    language_pref = st.session_state.preferences.get("language", "English")

    # Only show recommendations if adjusted_song_type exists (i.e. user confirmed mood improvement)
    if adjusted_song_type:
        if not st.session_state.current_songs or st.button("Refresh Recommendations"):
            if youtube is None:
                st.warning("YouTube API not configured. Add YOUTUBE_API_KEY to .env to enable song fetching.")
            else:
                with st.spinner("Fetching song recommendations..."):
                    st.session_state.current_songs = get_youtube_recommendations(adjusted_song_type or driving_emotion, song_type_pref, language_pref, limit=10)
                    st.session_state.playlist_url = generate_playlist_url(st.session_state.current_songs)

    if st.session_state.current_songs:
        for s in st.session_state.current_songs:
            c1, c2 = st.columns([3,1])
            with c1:
                st.markdown(f"**{s['name']}**")
                st.markdown(f"*{s['artist']}*")
            with c2:
                st.markdown(f"<a href='{s['link']}' target='_blank'>Open in YouTube</a>", unsafe_allow_html=True)
            st.write("---")
        if st.session_state.playlist_url:
            st.subheader("Your Emotion-Based Playlist")
            st.components.v1.html(f'<iframe width="100%" height="315" src="{st.session_state.playlist_url}" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>', height=340)

    # Reset button at bottom as well
    if st.button("Reset All (also clears session lists)"):
        st.session_state.emotion_list = []
        st.session_state.voice_label = None
        st.session_state.voice_common = None
        st.session_state.voice_conf = None
        st.session_state.voice_debug = None
        st.session_state.current_songs = []
        st.session_state.playlist_url = None
        st.session_state.mood_choice = None
        st.success("Reset complete")
        st.experimental_rerun()

# -------------------------
# Helper wrappers
# -------------------------
def get_top_from_scores(scores_dict):
    if not scores_dict:
        return None
    return max(scores_dict, key=scores_dict.get)

def prompt_for_mood_improvement_ui(dominant_emotion):
    suggested_type = MOOD_IMPROVEMENT_MAP.get(dominant_emotion)
    default = st.session_state.preferences.get("song_type", "Pop")
    if dominant_emotion == "Neutral":
        st.write("Your mood seems neutral. Would you like to listen to something specific?")
        mood_choice = st.radio("Choose a mood:", ("Happy","Sad"), key="neutral_mood_choice_ui")
        if st.button("Confirm Mood Choice"):
            st.session_state.mood_choice = mood_choice
            return mood_choice
        return None
    if suggested_type and dominant_emotion != "Happy":
        st.write(f"Your current mood is {dominant_emotion}. Would you like to uplift your mood with {suggested_type.lower()} songs?")
        uplift = st.radio("Uplift your mood?", ("Yes","No"), key=f"uplift_{dominant_emotion}_ui")
        if st.button("Confirm Uplift"):
            st.session_state.mood_choice = suggested_type if uplift=="Yes" else dominant_emotion
            return st.session_state.mood_choice
        return None
    return default

# -------------------------
# Router / Main
# -------------------------
def main():
    if not st.session_state.logged_in:
        page = st.sidebar.radio("Start", ["Login", "Register"])
        if page == "Register":
            show_register_page()
        else:
            show_login_page()
    else:
        if st.session_state.page == 'onboarding':
            show_onboarding_page()
        else:
            show_main_app()

if __name__ == "__main__":
    main()
