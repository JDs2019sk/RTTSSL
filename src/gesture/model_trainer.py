"""
Model Trainer Module
Provides functionality for training gesture recognition models using
both image datasets and real-time captured data.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import mediapipe as mp
import json
from datetime import datetime

class ModelTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.model = None
        self.labels = []
        self.training_data = []
        self.training_labels = []
        
    def train_from_images(self, dataset_path):
        """Train model using image dataset"""
        print("Loading dataset...")
        self._load_dataset(dataset_path)
        
        print("Creating and training model...")
        self._create_model()
        self._train_model()
        
        print("Saving model and labels...")
        self._save_model()
        
    def train_realtime(self, num_samples=100, num_classes=5):
        """Train model using real-time captured data"""
        cap = cv2.VideoCapture(0)
        
        for class_idx in range(num_classes):
            class_name = input(f"Enter name for class {class_idx + 1}: ")
            self.labels.append(class_name)
            
            print(f"Capturing samples for {class_name}...")
            samples_captured = 0
            
            while samples_captured < num_samples:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    # Extract landmarks
                    landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
                    self.training_data.append(landmarks)
                    self.training_labels.append(class_idx)
                    samples_captured += 1
                    
                    # Display progress
                    cv2.putText(frame, f"Samples: {samples_captured}/{num_samples}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                cv2.imshow("Capture", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
                    
        cap.release()
        cv2.destroyAllWindows()
        
        print("Creating and training model...")
        self._create_model()
        self._train_model()
        
        print("Saving model and labels...")
        self._save_model()
        
    def _load_dataset(self, dataset_path):
        """Load and process image dataset"""
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                self.labels.append(class_name)
                class_idx = len(self.labels) - 1
                
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                        
                    # Process image
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(image_rgb)
                    
                    if results.multi_hand_landmarks:
                        landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
                        self.training_data.append(landmarks)
                        self.training_labels.append(class_idx)
                        
    def _extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
        
    def _create_model(self):
        """Create neural network model"""
        self.model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(63,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.labels), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def _train_model(self):
        """Train the model"""
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test)
        )
        
        # Save training history
        self._save_history(history.history)
        
    def _save_model(self):
        """Save model and labels"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = os.path.join('models', f'model_{timestamp}.h5')
        self.model.save(model_path)
        
        # Save labels
        labels_path = os.path.join('models', f'labels_{timestamp}.txt')
        with open(labels_path, 'w') as f:
            for label in self.labels:
                f.write(f"{label}\n")
                
    def _save_history(self, history):
        """Save training history"""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join('logs', f'history_{timestamp}.json')
        
        with open(history_path, 'w') as f:
            json.dump(history, f)
