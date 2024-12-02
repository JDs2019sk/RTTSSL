# 🎓 Training Guide for RTTSSL

## Prerequisites

1. **Hardware Requirements:**
   - Webcam (for real-time training)
   - Good lighting conditions
   - Consistent background (preferably plain)
   - GPU recommended for faster training

2. **Software Setup:**
   - Python 3.8+
   - All dependencies installed (`pip install -r requirements.txt`)
   - Enough disk space for datasets

## Training Methods

### 1. 📸 Image-based Training

This method uses a pre-collected dataset of images.

#### Dataset Structure
```
datasets/
├── letters/           # For letter sign language
│   ├── A/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── B/
│   │   ├── image1.jpg
│   │   └── ...
├── words/            # For word signs
│   ├── HELLO/
│   │   ├── image1.jpg
│   │   └── ...
└── gestures/         # For general gestures
    ├── THUMBS_UP/
    │   ├── image1.jpg
    │   └── ...
```

#### Steps:
1. Create the dataset structure above
2. Add at least 50-100 images per class
3. Run training:
```bash
# For letters
python train.py --mode image --dataset datasets/letters

# For words
python train.py --mode image --dataset datasets/words

# For gestures
python train.py --mode image --dataset datasets/gestures
```

### 2. 🎥 Real-time Training

This method captures training data in real-time using your webcam.

#### Steps:
1. Run the training script with your desired options:
```bash
# Basic usage (3 gestures, 50 samples each)
python train.py --mode realtime --classes 3 --samples 50

# For more gestures
python train.py --mode realtime --classes 5 --samples 100
```

2. Follow the interactive process:
   - Enter names for each gesture class when prompted
   - For each gesture:
     - Show your hand to the camera
     - The system will automatically capture samples
     - A counter shows progress (e.g., "thumbs_up: 35/50")
     - Press ESC to skip to the next gesture if needed
   - The model will train automatically after all samples are collected

#### Training Tips:
- Keep your hand within the camera frame
- Move your hand slightly between samples for variety
- Ensure good lighting on your hand
- Use a plain background if possible
- You can see your hand detection in real-time

## Training Parameters

Customize your training with these parameters:

```bash
python train.py [options]

Options:
  --mode {image,realtime}    Training mode (required)
  --dataset PATH            Path to dataset (required for image mode)
  --samples NUMBER          Samples per class (default: 100)
  --classes NUMBER          Number of classes (default: 3)
```

## Best Practices

1. **Before Training:**
   - Test your camera with `test_camera.py`
   - Ensure good lighting
   - Use a plain background
   - Plan your gestures beforehand

2. **During Training:**
   - Keep your hand clearly visible
   - Vary hand position slightly
   - Watch the sample counter
   - Use ESC to skip if needed
   - Don't rush - quality matters

3. **After Training:**
   - Test each gesture
   - Check recognition accuracy
   - Retrain if needed
   - Back up your models

## Troubleshooting

### Common Issues:

1. **Camera Not Working:**
   - Run `test_camera.py` to verify camera
   - Check USB connection
   - Try a different USB port
   - Verify camera permissions

2. **Poor Recognition:**
   - Add more training samples
   - Improve lighting
   - Use a plain background
   - Vary hand positions
   - Retrain the model

3. **Training Crashes:**
   - Reduce number of samples
   - Close other applications
   - Check system resources
   - Update dependencies

## Model Files

After successful training, files are saved in the `models/` directory:
```
models/
├── model_YYYYMMDD_HHMMSS.h5    # Trained model
└── labels_YYYYMMDD_HHMMSS.txt  # Class labels
```

The timestamp in filenames helps track different training sessions.

## Additional Tips

1. **For Best Results:**
   - Train in the same conditions you'll use
   - Include slight variations in gestures
   - Use at least 50 samples per class
   - Test thoroughly after training

2. **When to Retrain:**
   - Poor recognition accuracy
   - Adding new gestures
   - Changing lighting conditions
   - Different camera setup

3. **Backup Your Models:**
   - Keep copies of good models
   - Save both model and label files
   - Document training parameters
