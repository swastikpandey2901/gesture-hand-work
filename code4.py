import pickle
import cv2
import mediapipe as mp
import numpy as np
import webbrowser
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import comtypes

# URLs with proper quotation marks
url1 = 'https://chat.openai.com/'
url2 = 'https://www.youtube.com/'
url3 = 'https://www.google.co.in/'
url4 = 'https://www.linkedin.com/in/swastik-pandey-3ba5a62a9/'
url5 = 'https://classroom.google.com/u/1/?pli=1'
add = 10.0  # Volume increment

# Initialize Core Audio APIs


# Load the hand gesture classification model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hands detection
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Mapping of gesture labels to actions
labels_dict = {0: 'A', 1: 'B', 2: 'c', 3: 'd', 4: 'E',}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # Image to draw
                hand_landmarks,  # Model output
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i, landmark in enumerate(hand_landmarks.landmark):
                x = landmark.x
                y = landmark.y

                x_.append(x)
                y_.append(y)

        for i, landmark in enumerate(hand_landmarks.landmark):
            x = landmark.x
            y = landmark.y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1, y1 = int(min(x_) * W), int(min(y_) * H)
        x2, y2 = int(max(x_) * W), int(max(y_) * H)

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        devices = AudioUtilities.GetSpeakers()

        # Perform actions based on predicted gesture
        cv2.putText(frame, predicted_character , (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
        if predicted_character == 'A':
            webbrowser.open_new_tab(url3)
        elif predicted_character == 'B':
            webbrowser.open_new_tab(url2)
        elif predicted_character == 'c':
            webbrowser.open_new_tab(url1)
        elif predicted_character == 'd':
            webbrowser.open_new_tab(url4)
        elif predicted_character == 'e':
            webbrowser.open_new_tab(url5)
            

    cv2.imshow('frame', frame)

    # Change waitKey parameter to break on 'q' press and reduce wait time
    if cv2.waitKey(2500) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
