import cv2
import mediapipe as mp
import config

class Dashboard:
    def __init__(self):
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def draw(self, frame, result, steering, throttle, brake, calibrated):
        # 1. Hand Landmarks
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Draw the hand landmarks using Task-based API
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

        # 2. Bottom Dashboard Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 380), (640, 480), config.CLR_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 3. Steering Arc
        cx, cy = 320, 430
        cv2.ellipse(frame, (cx, cy), (100, 100), 0, 180, 360, (50, 50, 50), 2)
        end_angle = 270 + (steering * 90)
        cv2.ellipse(frame, (cx, cy), (100, 100), 0, 270, end_angle, config.CLR_MAIN, 10)
        cv2.putText(frame, "STEER", (cx-25, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.CLR_TEXT, 1)

        # 4. Pedal Bars
        self._draw_bar(frame, (580, 350), throttle, config.CLR_ACCENT, "GAS")
        self._draw_bar(frame, (550, 350), brake, config.CLR_DANGER, "BRK")

        # 5. Status Badge
        badge_clr = config.CLR_ACCENT if result.hand_landmarks else config.CLR_DANGER
        txt = "SYSTEM ACTIVE" if result.hand_landmarks else "HANDS LOST"
        cv2.rectangle(frame, (20, 15), (150, 45), badge_clr, -1)
        cv2.putText(frame, txt, (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if not calibrated:
            cv2.putText(frame, "PRESS 'C' TO CALIBRATE", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    def _draw_bar(self, frame, pos, val, color, label):
        cv2.rectangle(frame, (pos[0], pos[1]-150), (pos[0]+20, pos[1]), (30, 30, 30), -1)
        h = int(val * 150)
        cv2.rectangle(frame, (pos[0], pos[1]-h), (pos[0]+20, pos[1]), color, -1)
        cv2.putText(frame, label, (pos[0]-5, pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw_calibration(self, frame, remaining):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (config.MAIN_WIDTH, config.MAIN_HEIGHT), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"GET READY: {int(remaining)+1}", (180, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, config.CLR_ACCENT, 5)
