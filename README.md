# 🤖 RTTSSL - Real-Time Translation System for Sign Language

RTTSSL is a Python-based system designed to visually translate letters, words and gestures in real time.

## 💡 Features

- Real-time gesture recognition and translation
- Sign language letter and word detection
- Facial Recognition
- Model training capabilities
- Recording method for your translations
- Performance monitoring and optimization

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

2. Test your GPU:

   ```bash
   python test_gpu.py
   ```

3. Train the gesture recognition model:

   ```bash
   python -m src.gesture.model_trainer
   ```

See [Training file](docs/TRAINING.md) for detailed training instructions.

4. Run the main program:
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

| Key | Function           | Description                                  |
| --- | ------------------ | -------------------------------------------- |
| `Q` | Quit               | Exit training mode                           |
| `N` | New Label          | Set a new label for recording                |
| `S` | Sample Recording   | Start/stop recording samples                 |
| `T` | Train Model        | Start model training (min 20, max 150/label) |
| `R` | Retrain Gesture    | Retrain a specific gesture from scratch      |
| `I` | Show Training Info | Show training information                    |

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
├── arduino config/       # arduino configuration folder
│   └── script.py         # arduino function
├── config/               # Configuration files for the project
│   └── configs.yaml      # Keybinds/configs file
├── datasets/             # Training and testing datasets
│   ├── gesture/          # Images to train gestures
│   ├── letter/           # Images to train letters
│   └── word/             # Images to train words
├── docs/                 # Documentation
│   └── TRAINING.md       # Training guide
├── logs/                 # Log files from training and execution
├── models/               # Trained models
├── recordings/           # Recorded video files
├── src/                  # Source code directory
│   ├── face/             # Face detection modules
│   ├── gesture/          # Gesture recognition modules
│   ├── mouse/            # Mouse control modules
│   └── utils/            # Utility functions
├── test_images/          # Images taken in camera test
├── main.py               # Main application entry point
├── requirements.txt      # Package dependencies
├── test_camera.py        # Camera testing utility
└── gpu_test.py           # GPU test for performance tracking (NVIDIA only)
```

#### Some folders will not be present, but they will be created by executing the program. [`data/ datasets/ logs/ models/ recordings/ faces/ test_images`]

<<<<<<< HEAD
## 🐋 Docker Support

You can run RTTSSL inside a Docker container. Note that camera access and GPU support require additional configuration.

### Prerequisites

- Docker installed on your system
- NVIDIA Container Toolkit (for GPU support)

### Build and Run

1. Build the Docker image:

```bash
docker build -t rttssl .
```

2. Run with CPU only (with data persistence):

```bash
docker run -it --rm \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/datasets:/app/datasets" \
  -v "$(pwd)/recordings:/app/recordings" \
  -v "$(pwd)/logs:/app/logs" \
  rttssl
```

3. Run with NVIDIA GPU support (with data persistence):

```bash
docker run -it --rm \
  --gpus all \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/datasets:/app/datasets" \
  -v "$(pwd)/recordings:/app/recordings" \
  -v "$(pwd)/logs:/app/logs" \
  rttssl
```

### Docker Notes

- Camera access requires passing the video device to the container
- GUI apps need X11 socket sharing and DISPLAY variable
- GPU support requires NVIDIA Container Toolkit
- Models and datasets can be persisted using Docker volumes

### Troubleshooting Docker

1. Allow X11 connections:

```bash
xhost +local:docker
```

2. Check camera permissions:

```bash
ls -l /dev/video0
sudo usermod -a -G video $USER
```

3. Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
=======
>>>>>>> 2c8a8179c3a9bf3022b4fc4fcbabf080632a248c

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Created by Joel Dias (PAP Project TIIGR 22/25)
