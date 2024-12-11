# RTTSSL Training Guide

This guide explains how to train RTTSSL recognition models using different modes and training methods.

## Table of Contents

- [Training Modes](#training-modes)
- [Training Methods](#training-methods)
  - [Real-time Training](#1-real-time-training)
  - [Dataset Training](#2-dataset-training)
- [Directory Structure](#directory-structure)
- [Image Requirements](#image-requirements)
- [Training Tips](#training-tips)
- [Troubleshooting](#troubleshooting)
- [Command-Line Arguments](#command-line-arguments)

## Training Modes

RTTSSL supports four training modes:

1. **Gesture Mode**: For training gestures (e.g., 👍, ✌️, etc.)

   - Ideal for simple gesture recognition
   - Recommended for commands and controls

2. **Letter Mode**: For training sign language alphabet

   - Optimized for individual letter recognition
   - Foundation for sign language communication

3. **Word Mode**: For training sign language words

   - Enables complete word recognition
   - Supports movement sequences

4. **Face Mode**: For training facial recognition
   - Detects and recognizes facial expressions
   - Complements gesture recognition

## Training Methods

### 1. Real-time Training

Uses webcam to capture samples in real-time.

#### How to Use:

```bash
# Method 1: With arguments (recommended)
python -m src.gesture.model_trainer --mode gesture --type realtime

# Method 2: Interactive
python -m src.gesture.model_trainer
# Then select mode and choose "Real-time Training"
```

#### Training Commands:

- `n`: Create/set new label
- `s`: Start/stop recording samples
- `t`: Train the model
- `i`: Show training information
- `r`: Retrain specific label
- `q`: Quit

### 2. Dataset Training

Uses pre-captured images organized in directories.

#### How to Use:

```bash
# Method 1: With arguments (recommended)
python -m src.gesture.model_trainer --mode letter --type dataset

# Method 2: Interactive
python -m src.gesture.model_trainer
# Then select mode and choose "Image Dataset Training"
```

## Directory Structure

```
datasets/
├── gesture/                # For gestures
│   ├── thumbs_up/         # One directory per gesture
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   ├── peace/
│   │   ├── img1.jpg
│
├── letter/                # For letters
│   ├── A/                # One directory per letter
│   │   ├── img1.jpg
│   ├── B/
│   │   ├── img1.jpg
│
└── word/                  # For words
    ├── hello/            # One directory per word
    │   ├── img1.jpg
    ├── thanks/
        ├── img1.jpg
```

## Image Requirements

1. **Supported Formats**:

   - JPG/JPEG
   - PNG

2. **Quality**:

   - Minimum resolution: 640x480
   - Good lighting
   - Sharp focus
   - Adequate contrast

3. **Content**:
   - Clearly visible gesture/letter
   - Well-positioned hand/face
   - Preferably neutral background
   - Avoid distracting objects

## Training Tips

1. **Sample Quantity**:

   - Minimum: 50 images per label
   - Ideal: 100-200 images per label
   - More samples = better accuracy

2. **Important Variations**:

   - Different angles
   - Various camera distances
   - Diverse lighting conditions
   - Different people/hands
   - Various backgrounds

3. **Best Practices**:

   - Maintain gesture/letter consistency
   - Vary position and rotation moderately
   - Include challenging cases
   - Backup your datasets

4. **Optimization**:
   - Start with few labels
   - Test frequently
   - Add more data for problematic labels
   - Use cross-validation

## Troubleshooting

1. **Low Accuracy**:

   - Add more training images
   - Check image quality
   - Increase training epochs
   - Reduce number of classes

2. **Detection Errors**:

   - Improve lighting
   - Adjust camera position
   - Check background
   - Use contrasting clothing

3. **Memory Issues**:

   - Reduce image resolution
   - Decrease batch size
   - Process fewer images at once
   - Free unused RAM

4. **Common Errors**:
   - "No features extracted": Check gesture visibility
   - "Directory not found": Verify folder structure
   - "Model not saved": Check write permissions

## Command-Line Arguments

The trainer supports the following command-line arguments:

```bash
python -m src.gesture.model_trainer [arguments]
```

### Required Arguments

None - if no arguments are provided, the trainer will run in interactive mode.

### Optional Arguments

| Argument | Values                              | Description            |
| -------- | ----------------------------------- | ---------------------- |
| `--mode` | `gesture`, `letter`, `word`, `face` | Training mode to use   |
| `--type` | `realtime`, `dataset`               | Training method to use |

### Examples

```bash
# Train letters using image dataset
python -m src.gesture.model_trainer --mode letter --type dataset

# Train gestures in real-time
python -m src.gesture.model_trainer --mode gesture --type realtime

# Train words using image dataset
python -m src.gesture.model_trainer --mode word --type dataset

# Train faces in real-time
python -m src.gesture.model_trainer --mode face --type realtime

# Interactive mode (will show menus)
python -m src.gesture.model_trainer
```

## Generated Files

After training, the following files are created in `models/`:

- `[mode]_model.h5`: Trained model
- `[mode]_labels.json`: Class labels
- `[mode]_training_data.npz`: Training data

## Additional Notes

- Model is automatically saved after training
- Previous backups are preserved
- You can mix training methods
- System automatically selects best model
- Use validation to assess quality
- Monitor overfitting/underfitting
