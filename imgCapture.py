import cv2 as cv
import os

def main():
    NUM_OF_SETS = 2
    NUM_OF_IMAGES = 100
    FILE_LETTERS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    DATA_DIR = "./data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    #start vidio capture
    cap = cv.VideoCapture(0)
    
    #create dir for the letter
    for i in range(NUM_OF_SETS):
        if not os.path.exists(DATA_DIR+'/'+FILE_LETTERS[i]):
            os.makedirs(DATA_DIR+'/'+FILE_LETTERS[i])

        while True:
            #get the frame
            ret, frame = cap.read()
            cv.imshow('Hand Capture', frame)
            if cv.waitKey(1) == ord('q'):
                break

        path = DATA_DIR+'/'+FILE_LETTERS[i]
        for j in range(NUM_OF_IMAGES):
            ret, frame = cap.read()
            cv.imshow('Hand Capture', frame)
            cv.imwrite(path+'/'+str(j+1)+'.jpg',frame)

if __name__ == "__main__":
    main()