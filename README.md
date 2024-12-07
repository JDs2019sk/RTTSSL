# 🤖 RTTSSL - Real-Time Translation System for Sign Language

RTTSSL is a Python-based system designed to visually translate letters, words and gestures in real time.

## 💡 Features

- Real-time gesture recognition and translation
- Sign language letter and word detection
- Facial Recognition
- Model training capabilities
- Recording method for your translations

## 🔤 Requirements

- Python 3.8-3.11 (TensorFlow compatibility)
- Webcam
- Required packages listed in `requirements.txt`
- GPU is recommended for faster training

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

## ⌨️ Default Keybinds

### Main Application Controls

| Key   | Function            | Description                                     |
| ----- | ------------------- | ----------------------------------------------- |
| `1`   | Letter Mode         | Switch to sign language letter translation mode |
| `2`   | Word Mode           | Switch to sign language word translation mode   |
| `3`   | Gesture Mode        | Switch to gesture recognition mode              |
| `M`   | Mouse Control       | Toggle hand gesture mouse control               |
| `F`   | Face Detection      | Enable face detection and recognition           |
| `E`   | Toggle Face Mode    | Cycle between mesh, iris, and recognition modes |
| `Tab` | Toggle FPS          | Show/hide FPS counter                           |
| `P`   | Performance Monitor | Show/hide performance statistics                |
| `R`   | Recording           | Start/stop recording                            |
| `H`   | Help Menu           | Show help and controls overlay                  |
| `Esc` | Exit                | Close the application                           |

### Training Mode Controls

| Key | Function           | Description                                    |
| --- | ------------------ | ---------------------------------------------- |
| `Q` | Quit               | Exit training mode                             |
| `N` | New Label          | Set a new label for recording                  |
| `S` | Sample Recording   | Start/stop recording samples                   |
| `T` | Train Model        | Start model training (min 100, max 1000/label) |
| `R` | Retrain Gesture    | Retrain a specific gesture from scratch        |
| `I` | Show Training Info | Show training information                      |

For detailed training instructions, see [Training Guide](docs/TRAINING.md).

### Camera Test Controls

| Key | Function     | Description             |
| --- | ------------ | ----------------------- |
| `Q` | Quit         | Exit camera test        |
| `S` | Save Image   | Save a test image       |
| `R` | Reset Camera | Reset camera connection |

## 📂 Project Structure

```
RTTSSL/
├── config/               # Configuration files for the project
│   └── keybinds.yaml     # keybinds/config file
├── data/                 # Raw and processed data storage
├── datasets/             # Training and testing datasets
├── docs/                 # Documentation
│   └── TRAINING.md       # Training guide
├── logs/                 # Log files from training and execution
├── models/               # Trained models
├── recordings/           # Recorded video files
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

## Created by Joel Dias (PAP Project TIIGR 22/25)
