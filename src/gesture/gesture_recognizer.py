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
import json
from collections import deque

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.model = None
        self.labels = []
        self.mode = "gesture"
        
        # Prediction smoothing with improved parameters
        self.prediction_history = deque(maxlen=10)  # Increased history
        self.confidence_threshold = 0.3  # Lowered threshold
        self.min_consecutive_predictions = 3  # Reduced required consecutive predictions
        
        # Add mean and std for normalization
        self.mean = None
        self.std = None
        
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and labels based on current mode"""
        model_path = os.path.join('models', f'{self.mode}_model.h5')
        labels_path = os.path.join('models', f'{self.mode}_labels.json')
        data_path = os.path.join('models', f'{self.mode}_training_data.npz')
        
        try:
            # Load labels first
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
                
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load normalization parameters from training data
            if os.path.exists(data_path):
                data = np.load(data_path)
                X = data['data']
                self.mean = X.mean(axis=0)
                self.std = X.std(axis=0)
            
            # Enable GPU acceleration if available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
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
                landmarks = self._extract_landmarks(hand_landmarks)
                
                # Get prediction if model is loaded
                if self.model is not None:
                    prediction = self._predict(landmarks)
                    if prediction:
                        translation = prediction
                    
                # Draw enhanced hand visualization
                self._draw_enhanced_hand(frame, hand_landmarks)
                
        return frame, translation
        
    def _extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks with improved normalization"""
        try:
            # Get all coordinates first
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            z_coords = [lm.z for lm in hand_landmarks.landmark]
            
            # Calculate bounding box
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            min_z, max_z = min(z_coords), max(z_coords)
            
            # Calculate ranges with epsilon to prevent division by zero
            eps = 1e-6
            x_range = max(max_x - min_x, eps)
            y_range = max(max_y - min_y, eps)
            z_range = max(max_z - min_z, eps)
            
            # Normalize coordinates relative to bounding box
            landmarks = []
            for landmark in hand_landmarks.landmark:
                # Normalize each coordinate to range [0, 1]
                norm_x = (landmark.x - min_x) / x_range
                norm_y = (landmark.y - min_y) / y_range
                norm_z = (landmark.z - min_z) / z_range
                
                landmarks.extend([norm_x, norm_y, norm_z])
            
            return np.array(landmarks)
            
        except Exception as e:
            print(f"Error extracting landmarks: {str(e)}")
            raise
            
    def _predict(self, landmarks):
        """Make prediction using the loaded model with improved smoothing"""
        try:
            # Reshape and normalize input data
            landmarks = landmarks.reshape(1, -1)
            
            # Apply same normalization as training
            if self.mean is not None and self.std is not None:
                landmarks = (landmarks - self.mean) / self.std
            
            # Get prediction probabilities
            prediction = self.model.predict(landmarks, verbose=0)
            
            # Get top 2 predictions and their confidences
            top2_idx = np.argsort(prediction[0])[-2:][::-1]
            confidences = prediction[0][top2_idx]
            
            # Check if the best prediction is significantly better than the second best
            if confidences[0] > self.confidence_threshold and (len(confidences) == 1 or confidences[0] - confidences[1] > 0.2):
                predicted_label = self.labels[top2_idx[0]]
                self.prediction_history.append(predicted_label)
                
                # For debugging
                if confidences[0] > 0.7:
                    print(f"High confidence prediction: {predicted_label} ({confidences[0]:.2f})")
                    if len(confidences) > 1:
                        print(f"Second best: {self.labels[top2_idx[1]]} ({confidences[1]:.2f})")
            
            # Check for consistent predictions
            if len(self.prediction_history) >= self.min_consecutive_predictions:
                recent_predictions = list(self.prediction_history)[-self.min_consecutive_predictions:]
                if all(x == recent_predictions[0] for x in recent_predictions):
                    return recent_predictions[0]
            
            return None
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
        
    def _draw_enhanced_hand(self, frame, landmarks):
        """Draw enhanced hand visualization"""
        # Draw landmarks with custom styles
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(255, 255, 255), thickness=1)
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
