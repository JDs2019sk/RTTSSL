# 🎓 Training Guide for RTTSSL

## Prerequisites

1. **Hardware Requirements:**
   - Webcam (for real-time training)
   - Good lighting conditions
   - Consistent background (preferably plain)
   - GPU recommended for faster training

2. **Software Setup:**
   - Python 3.8-3.11 (TensorFlow compatibility)
   - All dependencies installed (`pip install -r requirements.txt`)
   - Enough disk space for model storage

## Training Methods

### 1. 🎥 Real-Time Training (Recommended)

This method allows you to train the model in real-time using your webcam.

#### Steps:
1. Start the training module:
   ```bash
   python -m src.gesture.model_trainer
   ```

2. Use the following controls:
   - Press 'g' to switch to Gesture training mode
   - Press 'l' to switch to Letter training mode
   - Press 'w' to switch to Word training mode
   - Press 'n' to set a new label
   - Press 's' to start/stop recording samples
   - Press 't' to train the model (after collecting enough samples)
   - Press 'q' to quit

3. Training Process:
   1. Select your training mode (gesture/letter/word) using 'g', 'l', or 'w'
   2. Press 'n' and enter a label:
      - For gestures: descriptive name (e.g., "thumbs_up")
      - For letters: single letter (e.g., "A")
      - For words: complete word (e.g., "hello")
   3. Press 's' to start recording samples
   4. Move your hand slightly to capture different angles
   5. Press 's' again to stop recording
   6. Repeat for each item you want to recognize
   7. Press 't' to train the model once you have at least 100 samples

#### Training Tips:
- **For Gestures:**
  - Use distinct hand positions
  - Include variations in gesture orientation
  - Keep gestures simple and repeatable

- **For Letters:**
  - Follow standard sign language alphabet
  - Keep hand orientation consistent
  - Focus on clear finger positions

- **For Words:**
  - Use common sign language word gestures
  - Practice the motion before recording
  - Maintain consistent speed

#### Best Practices:
- Collect at least 100 samples per item
- Include variations in hand position and angle
- Ensure consistent lighting
- Keep your hand within camera frame
- Use distinct gestures/signs for better recognition
- Record samples from different distances

### 2. Model Management

The system automatically manages separate models for:
- Gestures (`models/gesture_model.h5`)
- Letters (`models/letter_model.h5`)
- Words (`models/word_model.h5`)

Each mode has its own:
- Model file (`.h5`)
- Labels file (`_labels.txt`)
- Training data (`.npz`)

You can switch between modes at any time during training, and the system will automatically:
1. Save the current mode's data
2. Load the appropriate model for the new mode
3. Continue training with the loaded model

### 2. 📸 Image Dataset Training

For training with a pre-collected image dataset.

#### Dataset Structure
```
datasets/
├── gestures/
│   ├── thumbs_up/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── peace/
│   │   ├── image1.jpg
│   │   └── ...
```

#### Steps:
1. Organize your images following the structure above
2. Run training:
   ```bash
   python -m src.gesture.model_trainer --mode image --dataset datasets/gestures
   ```

## 🔍 Troubleshooting

### Common Issues:

1. **Camera Not Found**
   - Run `python test_camera.py` to verify camera setup
   - Check if other applications are using the camera
   - Ensure camera permissions are enabled

2. **Poor Recognition**
   - Collect more training samples
   - Ensure varied hand positions in training
   - Check lighting conditions
   - Verify camera resolution settings

3. **Training Errors**
   - Ensure Python version compatibility (3.8-3.11)
   - Verify all dependencies are installed
   - Check for adequate disk space
   - Monitor system resources during training

4. **Model Not Saving**
   - Check write permissions in the models directory
   - Ensure adequate disk space
   - Verify path structure exists

## 📊 Model Performance

- The model automatically splits data into training and validation sets
- Early stopping prevents overfitting
- Training history is saved for analysis
- Test accuracy is displayed after training

## 🔄 Updating Models

- Models are saved with timestamps
- Previous models are preserved
- New training sessions create new model files
- Best performing models are automatically selected
