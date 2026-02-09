import vgamepad as vg
import numpy as np
import config

class RacingController:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        self.steering_val = 0.0
        self.throttle_val = 0.0
        self.brake_val = 0.0
        self.smoothed_steering = 0.0

    def update_steering(self, l_pos, r_pos, angle_offset):
        if l_pos and r_pos:
            dx = r_pos[0] - l_pos[0]
            dy = r_pos[1] - l_pos[1]
            raw_angle = np.degrees(np.arctan2(dy, dx))
            
            offset = angle_offset if angle_offset is not None else 0
            final_angle = raw_angle - offset
            
            # Wrap angle
            if final_angle > 180: final_angle -= 360
            if final_angle < -180: final_angle += 360
            
            # Normalize and Deadzone
            target = np.clip(final_angle / config.MAX_STEER_ANGLE, -1.0, 1.0)
            if abs(target) < 0.05: target = 0
            
            # Exponential Smoothing
            self.smoothed_steering = (target * config.SMOOTHING_FACTOR) + (self.smoothed_steering * (1 - config.SMOOTHING_FACTOR))
            self.steering_val = self.smoothed_steering
            return raw_angle
        else:
            # Auto-Center
            self.steering_val *= 0.8
            if abs(self.steering_val) < 0.01: self.steering_val = 0
            return None

    def set_pedals(self, throttle, brake):
        self.throttle_val = throttle
        self.brake_val = brake

    def send_to_gamepad(self):
        self.gamepad.left_joystick_float(x_value_float=float(self.steering_val), y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=float(self.throttle_val))
        self.gamepad.left_trigger_float(value_float=float(self.brake_val))
        self.gamepad.update()
