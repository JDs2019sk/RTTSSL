# 🤖 RTTSSL - Real-Time Translation System for Sign Language

RTTSSL is a Python-based system designed to visually translate letters, words and gestures in real time.

## 💡 Features

- Real-time gesture recognition and translation
- Sign language letter and word detection
- Interactive model training interface
- Hand landmark detection and tracking
- Advanced facial detection:
  - Face mesh detection
  - Eye iris tracking
  - Face recognition with name assignment
- Real-time model training capabilities
- Record your Translations
- Automatic model versioning and backup

## 🔤 Requirements

- Python 3.8-3.11 (TensorFlow compatibility)
- Webcam with good resolution
- Required packages listed in `requirements.txt`
- GPU recommended for faster training

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JDs2019sk/RTTSSL.git
   cd RTTSSL
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📚 Usage

1. Test your camera setup:
   ```bash
   python test_camera.py
   ```

2. Train the gesture recognition model:
   ```bash
   python -m src.gesture.model_trainer
   ```
   Follow the on-screen instructions for training. See `docs/TRAINING.md` for detailed guidance.

3. Run the main program:
   ```bash
   python main.py
   ```

## 📂 Project Structure

```
RTTSSL/
├── src/
│   ├── gesture/           # Gesture recognition modules
│   ├── face/             # Face detection modules
│   └── utils/            # Utility functions
├── models/               # Trained models
├── docs/                 # Documentation
│   └── TRAINING.md       # Training guide
├── test_camera.py        # Camera testing utility
└── requirements.txt      # Package dependencies
```

## 🔄 Training Process

1. **Camera Test**
   - Run `test_camera.py` to verify your setup
   - Check FPS and resolution
   - Test image capture functionality

2. **Model Training**
   - Interactive training interface
   - Real-time sample collection
   - Automatic model versioning
   - Progress tracking and validation

3. **Model Management**
   - Automatic saving of trained models
   - Version control with timestamps
   - Performance metrics tracking
   - Easy model updating

See `docs/TRAINING.md` for detailed training instructions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe for hand landmark detection
- TensorFlow and Keras teams
- OpenCV community
