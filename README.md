# 🤖 RTTSSL - Real-Time Translation System for Sign Language

RTTSSL is a Python-based system designed to visually translate letters, words and gestures in real time.

## 💡 Features

- Real-time gesture recognition and translation
- Sign language letter and word detection
- Hand-controlled mouse movement
- Advanced facial detection:
  - Face mesh detection
  - Eye iris tracking
  - Face recognition with name assignment
- Real-time model training capabilities
- Record you Translations

## 🔤 Requirements

- Python 3.8+
- Webcam
- Required packages listed in `requirements.txt`

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

Run the main program:

```bash
python main.py
```

## 🎮 Default Keybinds

| Key | Function                                                                 |
| --- | ------------------------------------------------------------------------ |
| 1   | Switch to letter translation mode                                        |
| 2   | Switch to word translation mode                                          |
| 3   | Switch to gesture translation mode                                       |
| M   | Enable/disable mouse control mode                                        |
| F   | Enable/disable face detection mode                                       |
| E   | Switch between face mesh and eye iris detection (in face detection mode) |
| N   | Assign name to detected face                                             |
| Tab | Toggle FPS display                                                       |
| P   | Toggle Performance Stats                                                 |
| R   | Start Recording                                                          |
| H   | Show help menu                                                           |
| Esc | Exit program                                                             |

## 📊 Training

### Image-based Training

```bash
python train.py --mode image --dataset path/to/dataset
```

### Real-time Training

```bash
python train.py --mode realtime
```

#### See [TRAINING](docs/TRAINING.md) for more information.

## 🔍 Troubleshooting

### Common Issues

1. Poor Recognition

   - Retrain with more diverse data
   - Check lighting conditions
   - Adjust confidence thresholds

2. Performance Issues

   - Lower camera resolution
   - Reduce FPS
   - Check system resources

3. Camera Problems

   - Verify camera permissions
   - Check USB connections
   - Update drivers

## 📁 Project Structure

```
RTTSSL/
├── main.py                # Main program entry point
├── config/                # Configuration files
├── src/                   # Source code
│   ├── gesture/           # Gesture recognition modules
│   ├── face/              # Face detection modules
│   ├── mouse/             # Mouse control modules
│   └── utils/             # Utility functions
├── models/                # Trained models
├── datasets/              # Training datasets
├── recordings/            # Recording Files
├── data/                  # Data files
│   └── face_names.json    # Save the names assigned to the faces
└── tests/                 # Unit tests
```

#### The folders `models/ datasets/ recordings/ data/ tests/` will be created when you use certain features of the program (or you can create them yourself)

## 📄 License

This project is under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📜 Created by Joel Dias (PAP Project TIIGR 22/25)
