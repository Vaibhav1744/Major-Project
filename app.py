
import numpy as np
import streamlit as st
import cv2
import sys
import random
import time
import logging
import os
sys.path.append('./libs')
from dotenv import load_dotenv
from googleapiclient.discovery import build
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import pymongo
from auth import register_user, login_user
from database import users_collection

# Global variables
cap = None
model = None

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YouTube API Setup
YOUTUBE_API_KEY = 'AIzaSyBFTWy7pUlldbenmXop0vNokQqOoq9aqDU'  # Replace with your API key
try:
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
except Exception as e:
    logger.error(f"YouTube API initialization failed: {e}")
    st.error("Failed to connect to YouTube API. Please check your API key.")

# Emotion Dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Mood Improvement Mapping
mood_improvement_map = {
    "Angry": "Calm",
    "Disgusted": "Happy",
    "Fearful": "Devotional",
    "Happy": "Happy",  # No change needed
    "Neutral": None,   # Will prompt user for Happy or Sad
    "Sad": "Happy",
    "Surprised": "Happy"
}

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'emotion_list' not in st.session_state:
    st.session_state.emotion_list = []
if 'recommendation_offset' not in st.session_state:
    st.session_state.recommendation_offset = 0
if 'current_songs' not in st.session_state:
    st.session_state.current_songs = []
if 'playlist_url' not in st.session_state:
    st.session_state.playlist_url = None
if 'mood_choice' not in st.session_state:
    st.session_state.mood_choice = None

# Login page
def show_login_page():
    st.markdown("<h2 style='text-align: center;'>Login to Emotion Music</h2>", unsafe_allow_html=True)
    
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
                if not user_data.get('preferences') or not user_data['preferences'].get('song_type'):
                    st.session_state.page = 'onboarding'
                else:
                    st.session_state.page = 'main'
                st.rerun()
            else:
                st.error(message)
    
    with col2:
        if st.button("Sign Up"):
            st.session_state.page = 'register'
            st.rerun()

# Registration page
def show_register_page():
    st.markdown("<h2 style='text-align: center;'>Create an Account</h2>", unsafe_allow_html=True)
    
    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Register"):
            if not email or not username or not password or not confirm_password:
                st.error("Please fill all fields")
                return
            
            if password != confirm_password:
                st.error("Passwords do not match")
                return
            
            success, message = register_user(email, username, password)
            if success:
                st.success(message)
                st.session_state.logged_in = True
                st.session_state.user = {"username": username, "email": email, "preferences": {}}
                st.session_state.page = 'onboarding'
                st.rerun()
            else:
                st.error(message)
    
    with col2:
        if st.button("Back to Login"):
            st.session_state.page = 'login'
            st.rerun()

# Onboarding page
def show_onboarding_page():
    st.markdown("<h2 style='text-align: center;'>Set Your Music Preferences</h2>", unsafe_allow_html=True)
    st.write(f"Welcome, {st.session_state.user['username']}! Let's set your music preferences.")
    
    song_type = st.selectbox("What type of songs do you prefer?", 
                            ["Devotional", "Romantic", "Indie", "Pop", "Rock"])
    
    language = st.selectbox("What language do you prefer for music?", 
                           ["Hindi", "English", "Others"])
    
    if st.button("Save Preferences"):
        preferences = {
            "song_type": song_type,
            "language": language
        }
        users_collection.update_one(
            {"username": st.session_state.user['username']},
            {"$set": {"preferences": preferences}}
        )
        st.session_state.user['preferences'] = preferences
        st.session_state.page = 'main'
        st.success("Preferences saved successfully!")
        st.rerun()

# Song Recommendation Function
def get_youtube_recommendations(emotion, song_type, language, limit=10, offset=0):
    """Fetch song recommendations from YouTube."""
    query = f"{emotion.lower()} {song_type.lower()} {language.lower()} music"
    try:
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=50,
            videoCategoryId="10",  # Music category
            pageToken=None if offset == 0 else None
        )
        response = request.execute()
        songs = [
            {
                'name': item['snippet']['title'],
                'artist': item['snippet']['channelTitle'],
                'link': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                'video_id': item['id']['videoId']
            }
            for item in response['items'][offset:offset+limit]
        ]
        random.shuffle(songs)
        return songs[:limit]
    except Exception as e:
        logger.error(f"Error fetching YouTube recommendations: {e}")
        return []

# Function to generate YouTube playlist URL
def generate_playlist_url(songs):
    """Generate a YouTube playlist URL from a list of song video IDs."""
    if not songs:
        return None
    video_ids = ','.join(song['video_id'] for song in songs)
    playlist_url = f"https://www.youtube.com/embed/{songs[0]['video_id']}?playlist={video_ids}&autoplay=1&loop=1"
    return playlist_url

# Mood Improvement Prompt
def prompt_mood_improvement(dominant_emotion):
    """Prompt user for mood improvement and return adjusted song type."""
    user_preferences = st.session_state.user.get('preferences', {})
    default_song_type = user_preferences.get('song_type', "Pop")
    suggested_type = mood_improvement_map.get(dominant_emotion)

    if dominant_emotion == "Neutral":
        st.write("Your mood seems neutral. Would you like to listen to something specific?")
        mood_choice = st.radio("Choose a mood:", ("Happy", "Sad"), key="neutral_mood_choice")
        if st.button("Confirm Mood"):
            st.session_state.mood_choice = mood_choice
            return mood_choice
        return None  # Wait for user confirmation

    if suggested_type and dominant_emotion != "Happy":
        st.write(f"Your current mood is {dominant_emotion}. Would you like to uplift your mood with {suggested_type.lower()} songs?")
        uplift = st.radio("Uplift your mood?", ("Yes", "No"), key=f"uplift_{dominant_emotion}")
        if st.button("Confirm Choice"):
            st.session_state.mood_choice = suggested_type if uplift == "Yes" else dominant_emotion
            return st.session_state.mood_choice
        return None  # Wait for user confirmation
    
    return default_song_type  # Default to user's preference if no change needed (e.g., Happy)

# Emotion Preprocessing
def get_dominant_emotion(emotion_list):
    """Return the most frequent emotion."""
    if not emotion_list:
        return None
    counter = Counter(emotion_list)
    return counter.most_common(1)[0][0]

# Model Definition
def create_emotion_model():
    """Create and load the CNN model for emotion detection."""
    try:
        if not os.path.exists('model.h5'):
            st.error("model.h5 file not found in project directory!")
            return None
        
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        model.load_weights('model.h5')
        logger.info("Model weights loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model.h5: {e}")
        st.error("Failed to load model.h5. Ensure it's in the project directory and compatible.")
        return None

# Initialize Model
cv2.ocl.setUseOpenCL(False)
model = None

# Main application page
def show_main_app():
    global cap, model

    user_preferences = st.session_state.user.get('preferences', {})
    song_type = user_preferences.get('song_type', "Pop")
    language = user_preferences.get('language', "English")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<h2 style='text-align: center; color: white;'><b>Emotion-Based Music Recommendation</b></h2>", unsafe_allow_html=True)
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.page = 'login'
            if 'cap' in globals() and cap is not None and cap.isOpened():
                cap.release()
            st.rerun()
    
    st.sidebar.write(f"**User:** {st.session_state.user['username']}")
    st.sidebar.write(f"**Song Type:** {song_type}")
    st.sidebar.write(f"**Language:** {language}")
    
    if st.sidebar.button("Change Preferences"):
        st.session_state.page = 'onboarding'
        st.rerun()
    
    if model is None:
        model = create_emotion_model()
        if model is None:
            st.error("Emotion model failed to initialize!")
            return
    
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        st.error(f"Failed to load {face_cascade_path}. Ensure OpenCV is installed correctly.")
        return
    
    st.subheader("Scan Your Emotion")
    snapshot_container = st.empty()
    multi_window_container = st.columns(3)

    if st.button('SCAN EMOTION'):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("No camera available. Please check your webcam connection.")
                return
                
        with st.spinner("Scanning emotions..."):
            emotion_list = []
            snapshots = []
            frame_count = 0
            
            while frame_count < 28:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Camera feed interrupted.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                frame_count += 1

                if len(faces) == 0:
                    cv2.putText(frame, "No Face Detected", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    try:
                        cropped_img = cv2.resize(roi_gray, (48, 48))
                        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
                        cropped_img = cropped_img.astype('float32') / 255.0
                        prediction = model.predict(cropped_img, verbose=0)
                        max_index = int(np.argmax(prediction))
                        emotion = emotion_dict[max_index]
                        emotion_list.append(emotion)
                        cv2.putText(frame, emotion, (x+20, y-60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        if len(snapshots) < 3 and random.random() < 0.3:
                            snapshot = frame.copy()
                            snapshots.append((snapshot, emotion))
                    except Exception as e:
                        logger.error(f"Prediction error: {e}")
                        continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                snapshot_container.image(frame_rgb, caption="Real-Time Emotion Detection", 
                                      use_container_width=True)
                time.sleep(0.05)

            for idx, (snapshot, emotion) in enumerate(snapshots):
                with multi_window_container[idx]:
                    st.image(cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB), 
                            caption=f"Snapshot: {emotion}", use_container_width=True)

            cap.release()
            
            if emotion_list:
                st.session_state.emotion_list = emotion_list
                st.session_state.recommendation_offset = 0
                st.session_state.current_songs = []
                st.session_state.playlist_url = None
                st.session_state.mood_choice = None
                st.success(f"Emotion scan complete! Detected {len(emotion_list)} emotions.")
            else:
                st.warning("No emotions detected. Please ensure your face is visible and well-lit.")

    # Mood Improvement and Recommendations
    if 'emotion_list' in st.session_state and st.session_state.emotion_list:
        dominant_emotion = get_dominant_emotion(st.session_state.emotion_list)
        st.markdown(f"<h4 style='text-align: center;'>Detected Emotion: {dominant_emotion}</h4>", unsafe_allow_html=True)
        
        # Prompt for mood improvement
        if st.session_state.mood_choice is None:
            adjusted_song_type = prompt_mood_improvement(dominant_emotion)
            if adjusted_song_type is None:  # Waiting for user input
                return
        else:
            adjusted_song_type = st.session_state.mood_choice

        st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended Songs</b></h5>", unsafe_allow_html=True)
        
        if not st.session_state.current_songs or st.button("Refresh Recommendations"):
            with st.spinner("Fetching song recommendations..."):
                st.session_state.current_songs = get_youtube_recommendations(
                    adjusted_song_type,  # Use adjusted type instead of dominant_emotion
                    song_type, 
                    language, 
                    limit=10, 
                    offset=st.session_state.recommendation_offset
                )
                st.session_state.playlist_url = generate_playlist_url(st.session_state.current_songs)
        
        if st.session_state.current_songs:
            for song in st.session_state.current_songs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"<h4>{song['name']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p><i>{song['artist']}</i></p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<a href='{song['link']}' target='_blank'>Open in YouTube</a>", unsafe_allow_html=True)
                st.write("---")
            
            # Embed YouTube Playlist
            if st.session_state.playlist_url:
                st.subheader("Your Emotion-Based Playlist")
                iframe_code = f"""
                <iframe width="100%" height="315" src="{st.session_state.playlist_url}" 
                frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
                """
                st.components.v1.html(iframe_code, height=315)
            
            if st.button("Suggest More"):
                with st.spinner("Fetching more songs..."):
                    st.session_state.recommendation_offset += 10
                    new_songs = get_youtube_recommendations(
                        adjusted_song_type,  # Use adjusted type
                        song_type, 
                        language, 
                        limit=10, 
                        offset=st.session_state.recommendation_offset
                    )
                    if new_songs:
                        st.session_state.current_songs = new_songs
                        st.session_state.playlist_url = generate_playlist_url(st.session_state.current_songs)
                        st.rerun()
                    else:
                        st.warning("No more songs available.")
                        st.session_state.recommendation_offset -= 10
        else:
            st.warning("No songs found. Try adjusting preferences or rescanning.")
    else:
        st.info("Please scan your emotion to get song recommendations.")

    if st.button("Reset"):
        st.session_state.emotion_list = []
        st.session_state.recommendation_offset = 0
        st.session_state.current_songs = []
        st.session_state.playlist_url = None
        st.session_state.mood_choice = None
        st.rerun()

# Main app structure
def main():
    st.set_page_config(
        page_title="Emotion Music",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if not st.session_state.logged_in:
        if st.session_state.page == 'register':
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
