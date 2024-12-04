# 🤖 RTTSSL - Real-Time Translation System for Sign Language

RTTSSL is a Python-based system designed to visually translate letters, words and gestures in real time.

## 💡 Features

- Real-time gesture recognition and translation
- Sign language letter and word detection
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

See [Training file](docs/TRAINING.md) for detailed training instructions.

3. Run the main program:
   ```bash
   python main.py
   ```

## 📂 Project Structure

```
RTTSSL/
├── config                # Configuration files for the project
│   └── keybinds.yaml     # keybinds/config file
├── data                  # Raw and processed data storage
├── datasets              # Training and testing datasets
├── docs/                 # Documentation
│   └── TRAINING.md       # Training guide
├── logs                  # Log files from training and execution
├── models/               # Trained models
├── recordings            # Recorded video files
├── src/                  # Source code directory
│   ├── gesture/          # Gesture recognition modules
│   ├── face/             # Face detection modules
│   └── utils/            # Utility functions
├── main.py               # Main application entry point
├── requirements.txt      # Package dependencies
├── test_camera.py        # Camera testing utility
└── train.py              # Model training script
```

#### Some folders will not be present, but they will be created by executing the program. [`data/ datasets/ logs/ models/ recordings/`]

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe for hand landmark detection
- TensorFlow and Keras teams
- OpenCV community
