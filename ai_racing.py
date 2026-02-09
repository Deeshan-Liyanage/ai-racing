import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import config 

# Gets Video Input
capture = cv2.VideoCapture(0)

# Loads the model
base_options = python.BaseOptions(model_asset_path=config.model)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
    )   

# Creates the hand landmarker 
landmarker = vision.HandLandmarker.create_from_options(options)

current_steering_angle = 0

if not capture.isOpened():
    print ("Cannot open camera")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the color to RGB
    convertColor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=convertColor)

    # Detects hands
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            wrist = hand[0]
            print(wrist.x,wrist.y)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

landmarker.close()
capture.release()
cv2.destroyAllWindows()