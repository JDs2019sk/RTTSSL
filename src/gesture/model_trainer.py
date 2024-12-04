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
        self.mp_drawing = mp.solutions.drawing_utils
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
        self.mode = "gesture"  # Default mode
        
        # Try to load existing model and labels
        self._load_existing_model()
        
    def set_mode(self, mode):
        """Set the training mode (gesture, letter, or word)"""
        if mode in ["gesture", "letter", "word"]:
            self.mode = mode
            # Reset model and load appropriate one for the mode
            self.model = None
            self.labels = []
            self.training_data = []
            self.training_labels = []
            self._load_existing_model()
            print(f"\nSwitched to {mode.upper()} training mode")
            if self.labels:
                print(f"Loaded existing {mode} labels: {self.labels}")
        else:
            print(f"Invalid mode: {mode}. Use 'gesture', 'letter', or 'word'")

    def _load_existing_model(self):
        """Load existing model and labels if they exist"""
        try:
            model_path = os.path.join('models', f'{self.mode}_model.h5')
            labels_path = os.path.join('models', f'{self.mode}_labels.txt')
            
            if os.path.exists(model_path) and os.path.exists(labels_path):
                # Load existing labels
                with open(labels_path, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines()]
                print(f"Loaded existing labels: {self.labels}")
                
                # Load existing model
                self.model = tf.keras.models.load_model(model_path)
                print("Loaded existing model")
                
                # Load existing training data if available
                data_path = os.path.join('models', f'{self.mode}_training_data.npz')
                if os.path.exists(data_path):
                    data = np.load(data_path)
                    self.training_data = list(data['data'])
                    self.training_labels = list(data['labels'])
                    print(f"Loaded {len(self.training_data)} existing training samples")
        except Exception as e:
            print(f"Note: No existing model found or error loading it: {e}")
            
    def train(self):
        """Start the training process"""
        cap = cv2.VideoCapture(0)
        recording = False
        frame_count = 0
        current_label = None
        
        print("\nTraining Controls:")
        print("- Press 'g' to switch to Gesture mode")
        print("- Press 'l' to switch to Letter mode")
        print("- Press 'w' to switch to Word mode")
        print("- Press 'n' to set a new label")
        print("- Press 's' to start/stop recording samples")
        print("- Press 't' to train the model")
        print("- Press 'q' to quit\n")
        print(f"Current mode: {self.mode.upper()}")
        if self.labels:
            print(f"Existing labels: {self.labels}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if recording and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks
                    landmarks = self._extract_landmarks(hand_landmarks)
                    
                    # Add to training data
                    self.training_data.append(landmarks)
                    # Store the index of the current label
                    label_idx = self.labels.index(current_label)
                    self.training_labels.append(label_idx)
                    frame_count += 1
                    
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                    )
            
            # Display status
            status = f"Mode: {self.mode.upper()} | "
            status += f"Recording: {recording}"
            if current_label:
                status += f" | Current Label: {current_label}"
            status += f" | Samples: {frame_count}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            cv2.imshow('Hand Training', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                self.set_mode("gesture")
            elif key == ord('l'):
                self.set_mode("letter")
            elif key == ord('w'):
                self.set_mode("word")
            elif key == ord('s'):
                if current_label is None:
                    print("[X] Please set a label first using 'n'")
                    continue
                recording = not recording
                if recording:
                    print(f"[+] Started recording samples for '{current_label}'")
                else:
                    print(f"[+] Stopped recording. Total samples: {frame_count}")
            elif key == ord('n'):
                prompt = "Enter new "
                if self.mode == "gesture":
                    prompt += "gesture label"
                elif self.mode == "letter":
                    prompt += "letter"
                else:
                    prompt += "word"
                label = input(f"\n{prompt}: ").strip()
                if label:
                    current_label = label
                    if label not in self.labels:
                        self.labels.append(label)
                        print(f"[+] Added new label: {label}")
                    print(f"[+] Set current label to: {label}")
                    print(f"[+] Current labels: {self.labels}")
            elif key == ord('t'):
                if len(self.training_data) < 100:
                    print("[X] Not enough samples. Please record at least 100 samples.")
                    continue
                    
                print(f"\n[*] Training {self.mode} model...")
                print(f"[*] Labels: {self.labels}")
                print(f"[*] Training data shape: {np.array(self.training_data).shape}")
                print(f"[*] Training labels shape: {np.array(self.training_labels).shape}")
                self._create_model()
                self._train_model()
                self._save_model()
                print("[+] Training completed!")
        
        cap.release()
        cv2.destroyAllWindows()
        
    def _extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks"""
        try:
            # Get all coordinates first
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            z_coords = [lm.z for lm in hand_landmarks.landmark]
            
            # Calculate bounding box
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            min_z, max_z = min(z_coords), max(z_coords)
            
            # Calculate ranges
            x_range = max_x - min_x
            y_range = max_y - min_y
            z_range = max_z - min_z
            
            # Prevent division by zero
            x_range = x_range if x_range != 0 else 1
            y_range = y_range if y_range != 0 else 1
            z_range = z_range if z_range != 0 else 1
            
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
        
    def _create_model(self):
        """Create neural network model"""
        try:
            # Don't create model if no labels
            if not self.labels:
                print("No labels defined yet. Model will be created when labels are added.")
                return

            print(f"\nCreating model for {len(self.labels)} classes: {self.labels}")
            
            # For single class, use binary classification with sigmoid
            is_binary = len(self.labels) == 1
            
            if self.model is None:
                print("Creating new model from scratch")
                self.model = models.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(63,), name='dense_1'),
                    layers.BatchNormalization(name='batch_norm_1'),
                    layers.Dropout(0.3, name='dropout_1'),
                    layers.Dense(64, activation='relu', name='dense_2'),
                    layers.BatchNormalization(name='batch_norm_2'),
                    layers.Dropout(0.3, name='dropout_2'),
                    layers.Dense(32, activation='relu', name='dense_3'),
                    layers.BatchNormalization(name='batch_norm_3'),
                    # Use sigmoid for binary, softmax for multiclass
                    layers.Dense(1 if is_binary else len(self.labels), 
                               activation='sigmoid' if is_binary else 'softmax',
                               name='output')
                ])
                
                self.model.compile(
                    optimizer='adam',
                    # Use binary crossentropy for single class
                    loss='binary_crossentropy' if is_binary else 'sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                print(f"Model created with {'binary' if is_binary else 'multiclass'} classification")
                self.model.summary()
                
            else:
                # If we're transitioning from binary to multiclass or adding more classes
                if (not is_binary and self.model.output_shape[1] == 1) or \
                   (not is_binary and len(self.labels) > self.model.output_shape[1]):
                    
                    print(f"Updating model for {len(self.labels)} classes")
                    
                    # Create a new model with the same architecture but different output layer
                    timestamp = int(time.time())
                    new_model = models.Sequential([
                        # First layer needs input_shape
                        layers.Dense(128, activation='relu', input_shape=(63,), 
                                   name=f'dense_1_{timestamp}'),
                        layers.BatchNormalization(name=f'batch_norm_1_{timestamp}'),
                        layers.Dropout(0.3, name=f'dropout_1_{timestamp}'),
                        layers.Dense(64, activation='relu', name=f'dense_2_{timestamp}'),
                        layers.BatchNormalization(name=f'batch_norm_2_{timestamp}'),
                        layers.Dropout(0.3, name=f'dropout_2_{timestamp}'),
                        layers.Dense(32, activation='relu', name=f'dense_3_{timestamp}'),
                        layers.BatchNormalization(name=f'batch_norm_3_{timestamp}'),
                        layers.Dense(len(self.labels), activation='softmax', 
                                   name=f'output_{timestamp}')
                    ])
                    
                    # Copy weights from old model to new model (except last layer)
                    print("Copying weights from old model to new model")
                    for old_layer, new_layer in zip(self.model.layers[:-1], new_model.layers[:-1]):
                        new_layer.set_weights(old_layer.get_weights())
                    
                    self.model = new_model
                    self.model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    print("Model updated successfully")
                    self.model.summary()
                    
        except Exception as e:
            print(f"Error creating model: {str(e)}")
            raise
        
    def _train_model(self):
        """Train the model"""
        try:
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            # Normalize training data
            X = (X - X.mean()) / X.std()
            
            # For binary classification, convert labels to binary format
            if len(self.labels) == 1:
                y = (y == 0).astype(np.float32)
            else:
                # For multiclass, ensure labels are proper indices
                print(f"Training multiclass model with labels: {self.labels}")
                print(f"Label distribution: {np.unique(y, return_counts=True)}")
            
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
            
            # Print predictions for test set
            predictions = self.model.predict(X_test)
            if len(self.labels) == 1:
                print("\nSample predictions (binary):")
                for i in range(min(5, len(predictions))):
                    print(f"True: {y_test[i]}, Pred: {predictions[i][0]:.2f}")
            else:
                print("\nSample predictions (multiclass):")
                for i in range(min(5, len(predictions))):
                    pred_idx = np.argmax(predictions[i])
                    print(f"True: {self.labels[int(y_test[i])]}, Pred: {self.labels[pred_idx]} ({predictions[i][pred_idx]:.2f})")
            
            # Save training history
            self._save_history(history.history)
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise
        
    def _save_model(self):
        """Save model, labels, and training data"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save model with mode prefix
            model_path = os.path.join('models', f'{self.mode}_model.h5')
            self.model.save(model_path)
            print(f"Model saved to: {model_path}")
            
            # Save labels with mode prefix
            labels_path = os.path.join('models', f'{self.mode}_labels.txt')
            with open(labels_path, 'w') as f:
                for label in self.labels:
                    f.write(f"{label}\n")
            print(f"Labels saved to: {labels_path}")
            
            # Save training data with mode prefix
            data_path = os.path.join('models', f'{self.mode}_training_data.npz')
            np.savez(data_path, 
                     data=np.array(self.training_data), 
                     labels=np.array(self.training_labels))
            print(f"Training data saved to: {data_path}")
            
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
