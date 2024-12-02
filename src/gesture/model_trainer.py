"""
Model Trainer Module
Provides functionality for training gesture recognition models using
both image datasets and real-time captured data.
"""

import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
import os
import mediapipe as mp
import json
from datetime import datetime
import time

class ModelTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Changed to False for better real-time performance
            max_num_hands=1,          # Only track one hand to avoid confusion
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.model = None
        self.labels = []
        self.training_data = []
        self.training_labels = []
        
    def train(self):
        """Start the training process"""
        print("\n=== Hand Gesture Training ===")
        print("Initializing camera and models...")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("[X] Error: Could not open camera")
            return
            
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 's' to start/stop recording samples")
        print("- Press 't' to train the model")
        print("- Press 'n' to set a new gesture label")
        print("\nCurrent gesture label: None")
        
        recording = False
        current_label = None
        frame_count = 0
        last_sample_time = time.time()
        sample_delay = 0.1  # 100ms between samples
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[X] Error reading frame")
                continue
                
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    
                    # Record samples if active
                    if recording and current_label is not None:
                        current_time = time.time()
                        if current_time - last_sample_time >= sample_delay:
                            landmarks = self._extract_landmarks(hand_landmarks)
                            self.training_data.append(landmarks)
                            self.training_labels.append(self.labels.index(current_label))
                            frame_count += 1
                            last_sample_time = current_time
                            
                            # Show progress
                            cv2.putText(frame, f"Samples: {frame_count}", (10, 70),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show recording status
            status = "Recording" if recording else "Not Recording"
            cv2.putText(frame, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0) if recording else (0, 0, 255), 2)
            
            # Show current label
            if current_label:
                cv2.putText(frame, f"Label: {current_label}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow('Hand Gesture Training', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                recording = not recording
                if recording:
                    print(f"[+] Started recording samples for '{current_label}'")
                else:
                    print(f"[+] Stopped recording. Total samples: {frame_count}")
            elif key == ord('n'):
                label = input("\nEnter new gesture label: ").strip()
                if label:
                    current_label = label
                    if label not in self.labels:
                        self.labels.append(label)
                    print(f"[+] Set current label to: {label}")
            elif key == ord('t'):
                if len(self.training_data) < 100:
                    print("[X] Not enough samples. Please record at least 100 samples.")
                    continue
                    
                print("\n[*] Training model...")
                self._create_model()
                self._train_model()
                self._save_model()
                print("[+] Training completed!")
        
        cap.release()
        cv2.destroyAllWindows()
        
    def _extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
        
    def _create_model(self):
        """Create neural network model"""
        try:
            self.model = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=(63,)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(len(self.labels), activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e:
            print(f"Error creating model: {str(e)}")
            raise
        
    def _train_model(self):
        """Train the model"""
        try:
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            # Normalize data
            X = (X - X.mean()) / X.std()
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Early stopping to prevent overfitting
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train with progress bar
            history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"\nTest accuracy: {test_acc:.2%}")
            
            # Save training history
            self._save_history(history.history)
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise
        
    def _save_model(self):
        """Save model and labels"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save model
            model_path = os.path.join('models', f'model_{timestamp}.h5')
            self.model.save(model_path)
            print(f"Model saved to: {model_path}")
            
            # Save labels
            labels_path = os.path.join('models', f'labels_{timestamp}.txt')
            with open(labels_path, 'w') as f:
                for label in self.labels:
                    f.write(f"{label}\n")
            print(f"Labels saved to: {labels_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
                
    def _save_history(self, history):
        """Save training history"""
        try:
            os.makedirs('logs', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_path = os.path.join('logs', f'history_{timestamp}.json')
            
            with open(history_path, 'w') as f:
                json.dump(history, f)
            print(f"Training history saved to: {history_path}")
            
        except Exception as e:
            print(f"Error saving history: {str(e)}")
            # Don't raise here as this is not critical

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
