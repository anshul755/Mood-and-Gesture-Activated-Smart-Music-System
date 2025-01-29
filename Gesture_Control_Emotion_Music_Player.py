import os
import cv2
import mediapipe as mp
from deepface import DeepFace
import time as t
import random
import numpy as np
import streamlit as st
from PIL import Image

hand_capture = mp.solutions.hands
hands = hand_capture.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5)
draw_option = mp.solutions.drawing_utils
faces = cv2.CascadeClassifier(cv2.samples.findFile("haarcascade_frontalface_default.xml"))

def detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    multifaces = faces.detectMultiScale(gray, 1.5, 4)
    pred = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
    print(f"Emotion Predictions: {pred[0]['dominant_emotion']}")
    dominant_emotion = pred[0]['dominant_emotion']

    for (x, y, w, h) in multifaces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (x - 1, y - 1), cv2.FONT_HERSHEY_COMPLEX, 1.1, (0, 255, 0), 2)

    return dominant_emotion

def playmusic(emotion):
    if emotion == "happy":
        random_no = str(random.randint(1, 3))
        file_path = f"D:/Studies/Python_ML/Project ML/{random_no}.mp3"
        if os.path.exists(file_path):
            with open(file_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
        else:
            st.error(f"Error: File {file_path} does not exist.")
    elif emotion == "neutral":
        random_no = str(random.randint(4, 6))
        file_path = f"D:/Studies/Python_ML/Project ML/{random_no}.mp3"
        if os.path.exists(file_path):
            with open(file_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
        else:
            st.error(f"Error: File {file_path} does not exist.")
    elif emotion == "surprise":
        random_no = str(random.randint(7, 9))
        file_path = f"D:/Studies/Python_ML/Project ML/{random_no}.mp3"
        if os.path.exists(file_path):
            with open(file_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
        else:
            st.error(f"Error: File {file_path} does not exist.")
    elif emotion == "sad":
        random_no = str(random.randint(10, 12))
        file_path = f"D:/Studies/Python_ML/Project ML/{random_no}.mp3"
        if os.path.exists(file_path):
            with open(file_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
        else:
            st.error(f"Error: File {file_path} does not exist.")
    elif emotion == "angry":
        random_no = str(random.randint(13, 15))
        file_path = f"D:/Studies/Python_ML/Project ML/{random_no}.mp3"
        if os.path.exists(file_path):
            with open(file_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
        else:
            st.error(f"Error: File {file_path} does not exist.")
    elif emotion == "fear":
        random_no = str(random.randint(16, 18))
        file_path = f"D:/Studies/Python_ML/Project ML/{random_no}.mp3"
        if os.path.exists(file_path):
            with open(file_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
        else:
            st.error(f"Error: File {file_path} does not exist.")

def detecthand(hand, res):
    for classification in res.multi_handedness:
        label = classification.classification[0].label
        score = classification.classification[0].score
        text = f"{label} {round(score, 2)}"

        wrist_landmark = hand.landmark[hand_capture.HandLandmark.WRIST]
        coords = tuple(np.multiply(np.array([wrist_landmark.x, wrist_landmark.y]), [640, 480]).astype(int))

        if label.lower() == "left":
            st.write("Left Hand So nothing can be detected exiting the program....")
            print("Left hand detected. Exiting program.")
            st.stop()

        return text, coords

    return None

st.title("Gesture-Controlled Music Player")
st.write("Use your hand gestures to control the music playback.")
stframe = st.empty()

camera_input = st.camera_input("Capture your gesture")

if camera_input:
    f = Image.open(camera_input)
    img = np.array(f)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.flip(img, 1)

    rgb_f = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    o_hands = hands.process(rgb_f)
    img.flags.writeable = True

    rgb_f = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    all_hands = o_hands.multi_hand_landmarks

    if all_hands:
        for hand in all_hands:
            draw_option.draw_landmarks(img, hand, hand_capture.HAND_CONNECTIONS)

            answer = detecthand(hand, o_hands)

            if answer:
                emotion = detector(img)
                if emotion:
                    playmusic(emotion)
                    t.sleep(1)
            else:
                st.write("No valid hand detected.")

    stframe.image(img, channels="BGR", use_container_width=True)