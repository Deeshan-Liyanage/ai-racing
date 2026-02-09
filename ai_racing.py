import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import config 
import time

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
lHand_pos = None
rHand_pos = None
angle_offset = None
raw_angle = None
calib_timer = 0

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
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            hand_label = handedness[0].category_name
            wrist = landmarks[0]

            if hand_label == "Left":
                lHand_pos = (wrist.x, wrist.y)
            elif hand_label == "Right" :
                rHand_pos = (wrist.x, wrist.y)

        if lHand_pos is not None and rHand_pos is not None:
            # Get the difference between the Hands
            dx = rHand_pos[0] - lHand_pos[0]
            dy = rHand_pos[1] - lHand_pos[1]

            # Calculate the angle
            radians = np.arctan2(dy,dx)
            raw_angle = np.degrees(radians)

            if angle_offset is not None:
                final_angle = raw_angle - angle_offset
                if final_angle > 180: final_angle -= 360

                if final_angle < -180: final_angle += 360

                current_steering_angle = final_angle

            else:
                current_steering_angle = raw_angle 

            print(f"Steering Angle: {current_steering_angle}")

                

    # Calibration Countdown Logic (Runs every frame)
    if calib_timer > 0:
        remaining = calib_timer - time.time()
        
        if remaining <= 0:
            if raw_angle is not None:
                angle_offset = raw_angle
                print(f"CALIBRATED! Offset set to: {angle_offset:.2f}")
            else:
                print("FAILED: No hands detected!")
            calib_timer = 0  # Stop the countdown
        else:
            # Visual feedback on screen
            cv2.putText(frame, f"CALIBRATING IN: {int(remaining)+1}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show the frame AFTER all drawing is done
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        calib_timer = time.time() + 3  # Set target time to 3 seconds from now
        print("Calibration starting in 3 seconds... Get ready!")

landmarker.close()
capture.release()
cv2.destroyAllWindows()