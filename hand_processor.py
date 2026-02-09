import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config
import time

class HandProcessor:
    def __init__(self):
        self.start_time = time.time()
        base_options = python.BaseOptions(model_asset_path=config.model)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        
        result = self.landmarker.detect_for_video(rgb_image, timestamp_ms)
        
        l_pos, r_pos = None, None
        
        if result.hand_landmarks:
            if len(result.hand_landmarks) == 2:
                # Sort by X coordinate for robust assignment
                sorted_hands = sorted(result.hand_landmarks, key=lambda h: h[0].x)
                l_pos = (sorted_hands[0][0].x, sorted_hands[0][0].y)
                r_pos = (sorted_hands[1][0].x, sorted_hands[1][0].y)
            else:
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name
                    wrist = landmarks[0]
                    if label == "Left": l_pos = (wrist.x, wrist.y)
                    elif label == "Right": r_pos = (wrist.x, wrist.y)
                    
        return result, l_pos, r_pos

    def close(self):
        self.landmarker.close()
