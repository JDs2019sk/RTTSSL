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
   - Press 'n' to set a new gesture label
   - Press 's' to start/stop recording samples
   - Press 't' to train the model (after collecting enough samples)
   - Press 'q' to quit

3. Training Process:
   1. Press 'n' and enter a name for your gesture (e.g., "thumbs_up")
   2. Press 's' to start recording samples
   3. Move your hand slightly to capture different angles
   4. Press 's' again to stop recording
   5. Repeat for each gesture you want to recognize
   6. Press 't' to train the model once you have at least 100 samples

#### Best Practices:
- Collect at least 100 samples per gesture
- Include variations in hand position and angle
- Ensure consistent lighting
- Keep your hand within camera frame
- Use distinct gestures for better recognition
- Record samples from different distances

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
