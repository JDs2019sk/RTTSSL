"""
Gesture Recognizer Module
Provides functionality for recognizing hand gestures and translating them
using MediaPipe Hands and a trained TensorFlow model.
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from collections import deque

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1  # Use more accurate model
        )
        self.model = None
        self.labels = []
        self.mode = "gesture"  # "gesture", "letter", or "word"
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.8
        self.min_consecutive_predictions = 3
        
        # Drawing styles
        self.drawing_styles = {
            'landmark_style': self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2),
            'connection_style': self.mp_drawing.DrawingSpec(
                color=(255, 255, 255), thickness=1)
        }
        
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and labels based on current mode"""
        model_path = os.path.join('models', f'{self.mode}_model.h5')
        labels_path = os.path.join('models', f'{self.mode}_labels.txt')
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            # Enable GPU acceleration if available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading model: {e}")
            
    def process_frame(self, frame):
        """Process frame and recognize gestures with improved detection"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        translation = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand landmarks with improved normalization
                landmarks = self._extract_landmarks(hand_landmarks, frame.shape)
                
                # Get prediction if model is loaded
                if self.model is not None:
                    prediction = self._predict(landmarks)
                    if prediction:
                        translation = prediction
                    
                # Draw enhanced hand visualization
                self._draw_enhanced_hand(frame, hand_landmarks)
                
        return frame, translation
        
    def _extract_landmarks(self, hand_landmarks, frame_shape):
        """Extract and normalize hand landmarks with improved normalization"""
        # Extract raw landmarks
        landmarks = []
        height, width = frame_shape[:2]
        
        # Get hand bounding box
        x_coords = [landmark.x * width for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * height for landmark in hand_landmarks.landmark]
        bbox_min_x, bbox_max_x = min(x_coords), max(x_coords)
        bbox_min_y, bbox_max_y = min(y_coords), max(y_coords)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        
        # Normalize landmarks relative to bounding box
        for landmark in hand_landmarks.landmark:
            # Normalize x, y coordinates
            norm_x = (landmark.x * width - bbox_min_x) / bbox_width
            norm_y = (landmark.y * height - bbox_min_y) / bbox_height
            
            # Add depth information
            norm_z = landmark.z
            
            landmarks.extend([norm_x, norm_y, norm_z])
            
        return np.array(landmarks).reshape(1, -1)
        
    def _predict(self, landmarks):
        """Make prediction using the loaded model with smoothing"""
        prediction = self.model.predict(landmarks, verbose=0)
        label_idx = np.argmax(prediction[0])
        confidence = prediction[0][label_idx]
        
        if confidence > self.confidence_threshold:
            predicted_label = self.labels[label_idx]
            self.prediction_history.append(predicted_label)
            
            # Check for consistent predictions
            if len(self.prediction_history) >= self.min_consecutive_predictions:
                recent_predictions = list(self.prediction_history)[-self.min_consecutive_predictions:]
                if all(x == recent_predictions[0] for x in recent_predictions):
                    return recent_predictions[0]
                    
        return None
        
    def _draw_enhanced_hand(self, frame, landmarks):
        """Draw enhanced hand visualization"""
        # Draw landmarks with custom styles
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self.drawing_styles['landmark_style'],
            connection_drawing_spec=self.drawing_styles['connection_style']
        )
        
        # Add depth visualization
        height, width = frame.shape[:2]
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            # Adjust circle size based on z-coordinate (depth)
            radius = max(1, min(5, int(5 * (1 + landmark.z))))
            cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
            
    def set_mode(self, mode):
        """Set recognition mode and reload model"""
        if mode in ["gesture", "letter", "word"]:
            self.mode = mode
            self._load_model()
            # Clear prediction history when changing modes
            self.prediction_history.clear()
