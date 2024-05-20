import os
import pickle
import cv2 as cv
import mediapipe as mp
import matplotlib.pyplot as plt

#load the model for hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

DATA_DIR = "./data"
data = []
lables = []

for dir in os.listdir(DATA_DIR):
    for img_dir in os.listdir(os.path.join(DATA_DIR,dir))[:1]:
        data_aux = []
        img = cv.imread(os.path.join(DATA_DIR,dir,img_dir))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        #detect hands in image
        result = hands.process(img_rgb)

        #draw the landmarks
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                for i in range(len(hand_landmark.landmark)):
                    x = hand_landmark.landmark[i].x
                    y = hand_landmark.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            lables.append(dir)

f = open('data.pickle', 'wb')
pickle.dump({'data':data, 'lables':lables}, f)
f.close()