import cv2
import config
import time
from hand_processor import HandProcessor
from controller import RacingController
from dashboard import Dashboard

def main():
    # Initialize components
    capture = cv2.VideoCapture(0)
    processor = HandProcessor()
    ctrl = RacingController()
    dash = Dashboard()

    # Window setup
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

    angle_offset = None
    calib_timer = 0
    raw_angle = 0

    print("AI Racing Wheel Starting...")

    while True:
        ret, frame = capture.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        
        # 1. Process AI Hand tracking
        result, l_pos, r_pos = processor.process_frame(frame)

        # 2. Update Controller Physics
        curr_raw = ctrl.update_steering(l_pos, r_pos, angle_offset)
        if curr_raw is not None: raw_angle = curr_raw
        
        ctrl.send_to_gamepad()

        # 3. Draw UI
        dash.draw(frame, result, ctrl.steering_val, ctrl.throttle_val, ctrl.brake_val, (angle_offset is not None))

        # 4. Handle Calibration
        if calib_timer > 0:
            remaining = calib_timer - time.time()
            if remaining <= 0:
                if result.hand_landmarks:
                    angle_offset = raw_angle
                    print(f"CALIBRATED! Offset: {angle_offset:.2f}")
                calib_timer = 0
            else:
                dash.draw_calibration(frame, remaining)

        # 5. Show Window
        cv2.imshow(config.WINDOW_NAME, cv2.resize(frame, (config.MINI_WIDTH, config.MINI_HEIGHT)))

        # 6. Keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('c'):
            calib_timer = time.time() + 3
            print("Calibration starting...")
        if key == ord('9'):
            ctrl.set_pedals(1.0, 0.0)
            print("ACCEL")
        if key == ord('0'):
            ctrl.set_pedals(0.0, 1.0)
            print("BRAKE")

    # Cleanup
    processor.close()
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()