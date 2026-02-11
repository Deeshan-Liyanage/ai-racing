# AI Racing Wheel 🏎️🤖

An AI-powered virtual racing controller that transforms your webcam into a high-precision steering wheel. This project uses MediaPipe hand tracking to calculate the steering angle between your hands and translates it into virtual XInput (Xbox 360) gamepad signals.

## ✨ Features

- **Gesture-Based Steering**: Steer by rotating your hands as if holding a physical wheel.
- **Virtual Gamepad Emulation**: Seamlessly works with any racing game that supports Xbox 360 controllers (via `vgamepad`).
- **Real-time Dashboard**: Integrated UI overlay showing:
    - Steering arc and angle visualization.
    - Throttle and Brake bars.
    - Hand tracking status and calibration prompts.
- **Calibration System**: Press a key to set your "neutral" center position.
- **Driving Physics**: Includes exponential smoothing and deadzone management for a stable driving experience.

## 🛠️ Requirements

- **OS**: Windows (Required for `vgamepad` / ViGEmBus)
- **Hardware**: Webcam
## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-racing
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Download AI Model**:
   Place the `hand_landmarker.task` file in the `model/` directory. You can download it from [MediaPipe's official documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models).

## 🎮 Usage

1. **Launch the application**:
   ```bash
   python ai_racing.py
   ```

2. **Calibration**:
   Once the window opens, hold your hands up in a comfortable "10 and 2" position and press **`C`**. A 3-second countdown will start to calibrate your neutral center.

3. **Controls**:
   - **Steer**: Rotate your hands left/right.
   - **Accelerate**: Press **`9`** on your keyboard (Mapped to Right Trigger).
   - **Brake**: Press **`0`** on your keyboard (Mapped to Left Trigger).
   - **Calibrate**: Press **`C`**.
   - **Quit**: Press **`Q`**.

## ⚙️ Configuration

You can tweak the driving feel in `config.py`:
- `MAX_STEER_ANGLE`: Adjust how much you need to rotate your hands for full lock (default: 45°).
- `SMOOTHING_FACTOR`: Increase for steadier steering, decrease for faster response.

## 📂 Project Structure

- `ai_racing.py`: Main entry point and application loop.
- `hand_processor.py`: MediaPipe hand detection and coordinate logic.
- `controller.py`: Virtual gamepad interface and steering physics.
- `dashboard.py`: OpenCV-based UI and dashboard drawing.
- `config.py`: Centralized settings and constants.
