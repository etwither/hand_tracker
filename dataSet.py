import os
import cv2 as cv
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

DATA_DIR = "./data"

for dir in os.listdir(DATA_DIR):
    for img_dir in os.listdir(os.path.join(DATA_DIR,dir))[:1]:
        img = cv.imread(os.path.join(DATA_DIR,dir,img_dir))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                #draw the landmarks
                mp_draw.draw_landmarks(img_rgb, hand_landmark, mp_hands.HAND_CONNECTIONS)

        plt.figure()
        plt.imshow(img_rgb)

plt.show()