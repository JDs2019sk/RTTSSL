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
2. Add at least 100 images per class
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
1. Run the training script:
```bash
python train.py --mode realtime
```

2. Follow the prompts:
   - Enter the name for each gesture/sign
   - Hold the gesture/sign steady
   - The system will capture multiple samples
   - Repeat for each gesture/sign you want to recognize

#### Training Tips:
- Vary the position slightly for each sample
- Vary the lighting conditions
- Include different backgrounds
- Capture from different angles

## Training Parameters

You can customize the training with these parameters:

```bash
python train.py [options]

Options:
  --mode {image,realtime}    Training mode
  --dataset PATH            Path to dataset (for image mode)
  --samples NUMBER          Samples per class (default: 100)
  --classes NUMBER          Number of classes (default: 5)
```

## Best Practices

1. **Data Collection:**
   - Use consistent lighting
   - Vary hand positions slightly
   - Include different backgrounds
   - Capture from different angles
   - At least 100 samples per class

2. **Model Training:**
   - Start with a small dataset to test
   - Gradually add more data
   - Monitor training accuracy
   - Use validation data
   - Save model checkpoints

3. **Testing:**
   - Test in different conditions
   - Verify accuracy
   - Check for false positives
   - Test edge cases

## Troubleshooting

### Common Issues:

1. **Poor Recognition:**
   - Add more training data
   - Improve lighting conditions
   - Ensure consistent hand positioning
   - Retrain with more diverse data

2. **Slow Training:**
   - Reduce image resolution
   - Use GPU if available
   - Reduce batch size
   - Optimize dataset size

3. **Overfitting:**
   - Add more training data
   - Use data augmentation
   - Adjust model complexity
   - Implement dropout

## Model Files

After training, the models are saved in the `models/` directory:
```
models/
├── gesture_model.h5    # Gesture recognition model
├── letter_model.h5     # Letter recognition model
├── word_model.h5       # Word recognition model
└── labels/            # Label files for each model
```
