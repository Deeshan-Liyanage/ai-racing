import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import config 
import time
import vgamepad as vg

# State initializations
current_steering_angle = 0
steering_val = 0
throttle_val = 0.0
brake_val = 0.0
angle_offset = None
raw_angle = 0
calib_timer = 0

# Gets Video Input
capture = cv2.VideoCapture(0)

# Start time for video mode timestamps
start_time = time.time()

# Loads the model
base_options = python.BaseOptions(model_asset_path=config.model)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.3, # Hold on to hands at sharp angles
    min_tracking_confidence=0.3,
)   

# Creates the hand landmarker 
landmarker = vision.HandLandmarker.create_from_options(options)

# Setup Window
cv2.namedWindow('AI Racing Wheel', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('AI Racing Wheel', cv2.WND_PROP_TOPMOST, 1) # Keep window on top

gamepad = vg.VX360Gamepad()

if not capture.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Reset positions at the start of each frame
    lHand_pos = None
    rHand_pos = None

    frame = cv2.flip(frame, 1)

    # Convert the color to RGB for MediaPipe
    convertColor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=convertColor)

    # Detects hands (VIDEO mode requires timestamp)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Draw landmarks on our BGR frame if hands are detected
    if result.hand_landmarks:
        frame = config.draw_landmarks_on_image(frame, result)
        
        # Robust Hand Assignment: 
        # If we see 2 hands, assume the one on the left of the screen is the "Left hand"
        if len(result.hand_landmarks) == 2:
            # Sort hands by X-coordinate
            sorted_hands = sorted(result.hand_landmarks, key=lambda landmarks: landmarks[0].x)
            lHand_pos = (sorted_hands[0][0].x, sorted_hands[0][0].y)
            rHand_pos = (sorted_hands[1][0].x, sorted_hands[1][0].y)
        else:
            # Fallback for 1 hand (still uses MediaPipe's label)
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                hand_label = handedness[0].category_name
                wrist = landmarks[0]
                if hand_label == "Left":
                    lHand_pos = (wrist.x, wrist.y)
                elif hand_label == "Right":
                    rHand_pos = (wrist.x, wrist.y)

        if lHand_pos is not None and rHand_pos is not None:
            # Calculate Steering Angle
            dx = rHand_pos[0] - lHand_pos[0]
            dy = rHand_pos[1] - lHand_pos[1]
            radians = np.arctan2(dy, dx)
            raw_angle = np.degrees(radians)

            # Use 0 if not calibrated yet, but still allow steering
            offset = angle_offset if angle_offset is not None else 0
            final_angle = raw_angle - offset
            
            if final_angle > 180: final_angle -= 360
            if final_angle < -180: final_angle += 360
            current_steering_angle = final_angle

            steering_val = np.clip(current_steering_angle / config.MAX_STEER_ANGLE, -1.0, 1.0)
        else:
            # Reset to center if one or both hands are lost
            steering_val = 0.0
    else:
        # Reset to center if no hands are detected
        steering_val = 0.0

    # Always update gamepad state
    gamepad.left_joystick_float(x_value_float=float(steering_val), y_value_float=0.0)
    gamepad.right_trigger_float(value_float=float(throttle_val))
    gamepad.left_trigger_float(value_float=float(brake_val))
    gamepad.update()

    # Visual Steering Bar
    bar_x = int(320 + steering_val * 200)
    cv2.line(frame, (320, 400), (bar_x, 400), (0, 255, 0), 10)
    cv2.circle(frame, (320, 400), 5, (255, 255, 255), -1)

    # Calibration Countdown Logic
    if calib_timer > 0:
        remaining = calib_timer - time.time()
        if remaining <= 0:
            if result.hand_landmarks:
                angle_offset = raw_angle
                print(f"CALIBRATED! Offset: {angle_offset:.2f}")
            else:
                print("FAILED: No hands detected for calibration!")
            calib_timer = 0
        else:
            cv2.putText(frame, f"CALIBRATING IN: {int(remaining)+1}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Status Overlay
    color = (0, 255, 0) if (lHand_pos and rHand_pos) else (0, 0, 255)
    cv2.putText(frame, f"Steer: {steering_val:.2f}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if angle_offset is None:
        cv2.putText(frame, "NOT CALIBRATED (Press 'C')", (50, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)


    ini_frame = cv2.resize(frame, (320, 240))
    cv2.imshow('AI Racing Wheel', ini_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        calib_timer = time.time() + 3
        print("Calibration starting in 3 seconds... Get ready!")
    if key == ord('9'):
        throttle_val = 1.0
        brake_val = 0.0
        print("ACCELERATING (9)")
    if key == ord('0'):
        throttle_val = 0.0
        brake_val = 1.0
        print("STOPPING (0)")

landmarker.close()
capture.release()
cv2.destroyAllWindows() 