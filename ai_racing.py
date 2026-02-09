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

# Smoothing variables
smoothed_steering = 0
SMOOTHING_FACTOR = 0.2 # 0.1 = very slow/smooth, 1.0 = instant/raw

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
            
            # 1. DEADZONE: Make straight-ahead driving easier
            target_steer = np.clip(final_angle / config.MAX_STEER_ANGLE, -1.0, 1.0)
            if abs(target_steer) < 0.05: target_steer = 0
            
            # 2. SMOOTHING: Prevent jittery movements
            smoothed_steering = (target_steer * SMOOTHING_FACTOR) + (smoothed_steering * (1 - SMOOTHING_FACTOR))
            steering_val = smoothed_steering
        else:
            # Gradually return to center if hands are lost
            steering_val = steering_val * 0.8
            if abs(steering_val) < 0.01: steering_val = 0

    # Always update gamepad state
    gamepad.left_joystick_float(x_value_float=float(steering_val), y_value_float=0.0)
    gamepad.right_trigger_float(value_float=float(throttle_val))
    gamepad.left_trigger_float(value_float=float(brake_val))
    gamepad.update()

    # --- ADVANCED RACING UI ---
    
    # 1. Background Overlay (Bottom Dashboard)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 380), (640, 480), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 2. Steering Arc Gauge
    center_x, center_y = 320, 430
    cv2.ellipse(frame, (center_x, center_y), (100, 100), 0, 180, 360, (50, 50, 50), 2) # Background
    # Draw Active Arc
    steer_angle = steering_val * 90 
    end_angle = 270 + steer_angle
    cv2.ellipse(frame, (center_x, center_y), (100, 100), 0, 270, end_angle, (0, 255, 255), 10)
    cv2.putText(frame, "STEER", (center_x-25, center_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 3. Throttle & Brake Vertical "LED" Bars
    # Throttle (Right)
    cv2.rectangle(frame, (580, 200), (600, 350), (30, 30, 30), -1) # Background
    th_h = int(throttle_val * 150)
    cv2.rectangle(frame, (580, 350 - th_h), (600, 350), (0, 255, 0), -1)
    cv2.putText(frame, "GAS", (575, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Brake (Left of Throttle)
    cv2.rectangle(frame, (550, 200), (570, 350), (30, 30, 30), -1) # Background
    br_h = int(brake_val * 150)
    cv2.rectangle(frame, (550, 350 - br_h), (570, 350), (0, 0, 255), -1)
    cv2.putText(frame, "BRK", (545, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 4. Status Badges (Top Left)
    status_y = 40
    if lHand_pos and rHand_pos:
        cv2.rectangle(frame, (20, status_y-25), (150, status_y+5), (0, 150, 0), -1)
        cv2.putText(frame, "SYSTEM ACTIVE", (30, status_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (20, status_y-25), (150, status_y+5), (0, 0, 150), -1)
        cv2.putText(frame, "HANDS LOST", (30, status_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Calibration Overlay
    if calib_timer > 0:
        remaining = calib_timer - time.time()
        if remaining <= 0:
            if result.hand_landmarks:
                angle_offset = raw_angle
                print(f"CALIBRATED! Offset: {angle_offset:.2f}")
            else:
                print("FAILED: No hands detected!")
            calib_timer = 0
        else:
            # Full screen darken for calibration focus
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (640,480), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, f"GET READY: {int(remaining)+1}", (180, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)

    if angle_offset is None:
        cv2.putText(frame, "PRESS 'C' TO CALIBRATE", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # --- END UI ---


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