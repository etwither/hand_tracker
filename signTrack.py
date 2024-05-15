import cv2 as cv
import argparse
import mediapipe as mp

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument("--min_detection_confidence", help="min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help="min_tracking_confidence", type=int, default=0.5)

    args = parser.parse_args()

    return args

def main():
    #set arguments
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    #camera setting
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    #load the model for hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=args.use_static_image_mode, max_num_hands=2, min_detection_confidence=args.min_detection_confidence, min_tracking_confidence=args.min_tracking_confidence)
    mp_draw = mp.solutions.drawing_utils

    while True:
        _,frame = cap.read()

        frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        #place the landmarks in the image
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame_rgb, hand_landmark, mp_hands.HAND_CONNECTIONS)

        cv.imshow("capture", frame_rgb)

        #wait for 'q' to be pressed
        if cv.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()