"""
Model Trainer Module
Provides functionality for training gesture recognition models using
both image datasets and real-time captured data.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import mediapipe as mp
import json
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
import yaml
import argparse

class ModelTrainer:
    def __init__(self, mode=None, training_type=None):
        self.mode = mode
        self.training_type = training_type
        
        if not mode:
            print("\nWelcome to RTTSSL Training!")
            print("\nSelect training mode:")
            print("1. Gesture Mode")
            print("2. Letter Mode")
            print("3. Word Mode")
            print("4. Face Mode")
            
            while True:
                try:
                    choice = input("\nEnter mode (1-4): ").strip()
                    if choice == '1':
                        self.mode = "gesture"
                        break
                    elif choice == '2':
                        self.mode = "letter"
                        break
                    elif choice == '3':
                        self.mode = "word"
                        break
                    elif choice == '4':
                        self.mode = "face"
                        break
                    else:
                        print("Invalid choice. Please enter 1, 2, 3, or 4.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        if not training_type:
            print("\nSelect training type:")
            print("1. Real-time Training (using webcam)")
            print("2. Image Dataset Training (using saved images)")
            
            while True:
                try:
                    choice = input("\nEnter training type (1-2): ").strip()
                    if choice == '1':
                        self.training_type = "realtime"
                        break
                    elif choice == '2':
                        self.training_type = "dataset"
                        break
                    else:
                        print("Invalid choice. Please enter 1 or 2.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        print(f"\nInitializing {self.mode.upper()} training mode with {self.training_type.upper()} training...")
        
        # Load keybinds configuration
        self.config = self._load_config()
        self.keybinds = self.config['keybinds']
        self.training_config = self.config.get('training', {})
        
        # Initialize MediaPipe based on mode
        if self.mode == "face":
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=self.training_config.get('confidence_threshold', 0.5),
                model_selection=1
            )
        else:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.training_type == "dataset",
                max_num_hands=2,
                min_detection_confidence=self.training_config.get('confidence_threshold', 0.5),
                min_tracking_confidence=self.training_config.get('confidence_threshold', 0.5),
                model_complexity=self.training_config.get('model_complexity', 1)
            )
            
        self.model = None
        self.labels = []
        self.samples_per_label = {}
        
        # Try to load existing model and labels
        self._load_existing_model()

    def _load_config(self):
        """Load configuration from YAML file"""
        config_path = os.path.join('config', 'keybinds.yaml')
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {
                'keybinds': {
                    'quit_training': 'q',
                    'start_stop_recording': 's',
                    'new_label': 'n',
                    'retrain_label': 'r',
                    'info_display': 'i',
                    'start_training': 't'
                },
                'training': {
                    'max_samples_per_label': 1000,
                    'window_size': {'width': 1280, 'height': 720},
                    'confidence_threshold': 0.5,
                    'model_complexity': 1
                }
            }
            
    def _load_existing_model(self):
        """Load existing model and labels if they exist"""
        try:
            model_path = os.path.join('models', f'{self.mode}_model.h5')
            labels_path = os.path.join('models', f'{self.mode}_labels.json')
            data_path = os.path.join('models', f'{self.mode}_training_data.npz')
            
            if os.path.exists(model_path) and os.path.exists(labels_path):
                self.model = keras.models.load_model(model_path)
                with open(labels_path, 'r') as f:
                    self.labels = json.load(f)
                
                if os.path.exists(data_path):
                    data = np.load(data_path)
                    self.samples_per_label = {label: [] for label in self.labels}
                    
                    # Convert label indices back to label names
                    for sample, label_idx in zip(data['data'], data['labels']):
                        label = self.labels[label_idx]
                        self.samples_per_label[label].append(sample)
                        
                    print(f"Loaded existing model with labels: {self.labels}")
                    for label, samples in self.samples_per_label.items():
                        print(f"  - {label}: {len(samples)} samples")
        except Exception as e:
            print(f"Note: No existing model found or error loading it: {e}")
            
    def _create_model(self, input_shape):
        """
        Create a new model for training.
        
        Args:
            input_shape (int): Number of input features
        """
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_from_images(self):
        """Train the model using images from the dataset directory."""
        print(f"\nFound {len(self.labels)} labels: {self.labels}")
        
        features = []
        labels = []
        label_indices = {label: idx for idx, label in enumerate(self.labels)}
        
        dataset_dir = os.path.join('datasets', self.mode)
        if not os.path.exists(dataset_dir):
            print(f"\nError: Dataset directory '{dataset_dir}' not found!")
            print("Please create the directory and add your image data.")
            return
        
        for label in self.labels:
            label_dir = os.path.join(dataset_dir, label)
            if not os.path.exists(label_dir):
                print(f"\nWarning: Directory for label '{label}' not found at {label_dir}")
                continue
                
            print(f"\nProcessing {label}...")
            image_files = [f for f in os.listdir(label_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(image_files)} images")
            
            valid_features = []
            for image_file in image_files:
                image_path = os.path.join(label_dir, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(image)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Extract x, y coordinates from landmarks
                            coords = []
                            for landmark in hand_landmarks.landmark:
                                coords.extend([landmark.x, landmark.y])
                            valid_features.append(coords)
                            labels.append(label_indices[label])
                            
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
                    continue
            
            if valid_features:
                features.extend(valid_features)
                print(f"Successfully extracted features from {len(valid_features)} images")
            else:
                print(f"No valid features extracted for label {label}")
        
        if not features:
            print("\nError: No valid features extracted from any images!")
            return
            
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        self.model = self._create_model(X_train.shape[1])
        
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        # Save model and labels
        self._save_model()
        print("\nModel saved successfully!")

    def train(self):
        """Start the training process based on selected type"""
        if self.training_type == "dataset":
            self.train_from_images()
        else:
            self.train_realtime()

    def train_realtime(self):
        """Train the model using real-time webcam capture"""
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            # Set higher resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            recording = False
            frame_count = 0
            current_label = None
            
            print("\nTraining Controls:")
            print(f"- Press '{self.keybinds['quit_training']}' to quit")
            print(f"- Press '{self.keybinds['start_stop_recording']}' to start/stop recording samples")
            print(f"- Press '{self.keybinds['new_label']}' to set/create a new label")
            print(f"- Press '{self.keybinds['retrain_label']}' to retrain a specific label from scratch")
            print(f"- Press '{self.keybinds['info_display']}' to show training info")
            print(f"- Press '{self.keybinds['start_training']}' to train the model")
            print("\n")
            
            print(f"Current mode: {self.mode.upper()}")
            if self.labels:
                print("Current labels and samples:")
                for label in self.labels:
                    count = len(self.samples_per_label.get(label, []))
                    print(f"  - {label}: {count} samples")
            
            while cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame based on mode
                    if self.mode == "face":
                        results = self.face_detection.process(frame_rgb)
                        if results.detections:
                            for detection in results.detections:
                                self.mp_drawing.draw_detection(frame, detection)
                    else:
                        results = self.hands.process(frame_rgb)
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                self.mp_drawing.draw_landmarks(
                                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Status display
                    status = f"Mode: {self.mode.upper()}"
                    if current_label:
                        status += f" | Label: {current_label}"
                        if current_label in self.samples_per_label:
                            status += f" ({len(self.samples_per_label[current_label])} samples)"
                    if recording:
                        status += " | RECORDING"
                        status += f" | Frame: {frame_count}"
                    
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    if recording and current_label:
                        # Extract features based on mode
                        features = self._extract_landmarks(results, frame if self.mode == "face" else None)
                        if features is not None:
                            if current_label not in self.samples_per_label:
                                self.samples_per_label[current_label] = []
                            self.samples_per_label[current_label].append(features)
                            frame_count += 1
                    
                    cv2.namedWindow('Hand Training')
                    window_size = self.training_config.get('window_size', {'width': 1280, 'height': 720})
                    cv2.resizeWindow('Hand Training', window_size['width'], window_size['height'])
                    cv2.imshow('Hand Training', frame)
                    
                    # Check if window is still open
                    if cv2.getWindowProperty('Hand Training', cv2.WND_PROP_VISIBLE) < 1:
                        break
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(self.keybinds['quit_training']):
                        break
                    elif key == ord(self.keybinds['start_stop_recording']):
                        if current_label:
                            # Check if maximum number of samples has been reached
                            if current_label in self.samples_per_label and len(self.samples_per_label[current_label]) >= self.training_config.get('max_samples_per_label', 1000):
                                print("\n[X] Maximum of 1000 samples reached for this label")
                                continue
                                
                            recording = not recording
                            if not recording:
                                print(f"\n[+] Recorded {frame_count} frames for {current_label}")
                                frame_count = 0
                        else:
                            print("\n[X] Please set a label first using 'n'")
                    elif key == ord(self.keybinds['new_label']):
                        # Destroy window before input to prevent focus issues
                        cv2.destroyWindow('Hand Training')
                        
                        prompt = "Enter "
                        if self.mode == "gesture":
                            prompt += "gesture label"
                        elif self.mode == "letter":
                            prompt += "letter"
                        elif self.mode == "face":
                            prompt += "face label"
                        else:
                            prompt += "word"
                        label = input(f"\n{prompt}: ").strip()
                        
                        # Recreate window after input
                        cv2.namedWindow('Hand Training')
                        window_size = self.training_config.get('window_size', {'width': 1280, 'height': 720})
                        cv2.resizeWindow('Hand Training', window_size['width'], window_size['height'])
                        
                        if label:
                            current_label = label
                            if label not in self.labels:
                                self.labels.append(label)
                                self.samples_per_label[label] = []
                                print(f"[+] Added new label: {label}")
                            print(f"[+] Set current label to: {label}")
                            print(f"[+] Current labels: {self.labels}")
                    elif key == ord(self.keybinds['retrain_label']):
                        if not self.labels:
                            print("\n[X] No labels available to retrain")
                            continue
                        
                        # Destroy window before input
                        cv2.destroyWindow('Hand Training')
                        
                        print("\nAvailable labels to retrain:")
                        for i, label in enumerate(self.labels):
                            count = len(self.samples_per_label.get(label, []))
                            print(f"{i + 1}. {label} ({count} samples)")
                        
                        try:
                            choice = input("\nEnter the number of the label to retrain (or 0 to cancel): ")
                            
                            # Recreate window after input
                            cv2.namedWindow('Hand Training')
                            window_size = self.training_config.get('window_size', {'width': 1280, 'height': 720})
                            cv2.resizeWindow('Hand Training', window_size['width'], window_size['height'])
                            
                            if choice.isdigit():
                                choice = int(choice)
                                if choice == 0:
                                    print("Cancelled retraining")
                                    continue
                                if 1 <= choice <= len(self.labels):
                                    label = self.labels[choice - 1]
                                    print(f"\n[+] Clearing samples for {label}")
                                    self.samples_per_label[label] = []
                                    current_label = label
                                    recording = False
                                    frame_count = 0
                                    print(f"[+] Ready to record new samples for {label}")
                                    print("[+] Press 's' to start recording")
                                else:
                                    print("[X] Invalid choice")
                            else:
                                print("[X] Please enter a number")
                        except Exception as e:
                            print(f"[X] Error during retrain: {e}")
                    elif key == ord(self.keybinds['info_display']):
                        print("\nTraining Information:")
                        print(f"Current mode: {self.mode.upper()}")
                        print("Labels and samples:")
                        for label in self.labels:
                            count = len(self.samples_per_label.get(label, []))
                            print(f"  - {label}: {count} samples")
                    elif key == ord(self.keybinds['start_training']):
                        total_samples = sum(len(samples) for samples in self.samples_per_label.values())
                        if total_samples < 100:
                            print("[X] Not enough samples. Please record at least 100 samples in total.")
                            continue
                        
                        # Check if each label has at least some samples
                        min_samples = min(len(samples) for samples in self.samples_per_label.values())
                        if min_samples < 20:
                            print("[X] Each label needs at least 20 samples.")
                            print("\nSamples per label:")
                            for label in self.labels:
                                count = len(self.samples_per_label[label])
                                print(f"  - {label}: {count} samples")
                            continue
                        
                        print(f"\n[*] Training {self.mode} model...")
                        print(f"[*] Labels: {self.labels}")
                        print("\nSamples per label:")
                        for label in self.labels:
                            count = len(self.samples_per_label[label])
                            print(f"  - {label}: {count} samples")
                        
                        self._train_model()
                        print("[+] Training completed!")
                
                except KeyboardInterrupt:
                    print("\n[!] Training interrupted by user")
                    break
                except Exception as e:
                    print(f"\n[X] Error during training: {e}")
                    continue
            
        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user")
        except Exception as e:
            print(f"\n[X] Error initializing training: {e}")
        finally:
            print("\n[*] Cleaning up...")
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            
            # Save current progress if there are samples
            if any(len(samples) > 0 for samples in self.samples_per_label.values()):
                try:
                    print("[*] Saving current progress...")
                    self._save_model()
                    print("[+] Progress saved successfully")
                except Exception as e:
                    print(f"[X] Error saving progress: {e}")
        
    def _extract_landmarks(self, detection_result, frame=None):
        """Extract and normalize landmarks based on mode"""
        try:
            if self.mode == "face":
                if not detection_result.detections:
                    return None
                    
                # Get face detection
                detection = detection_result.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                height, width = frame.shape[:2]
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Extract face region and resize to standard size
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    return None
                    
                face_img = cv2.resize(face_img, (128, 128))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
                # Normalize pixel values
                face_features = face_img.flatten() / 255.0
                return face_features
                
            else:
                # Original hand landmark extraction
                if not detection_result.multi_hand_landmarks:
                    return None
                    
                hand_landmarks = detection_result.multi_hand_landmarks[0]
                
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
        
    def _train_model(self):
        """Train the model"""
        try:
            # Prepare training data
            X = []
            y = []
            for label_idx, label in enumerate(self.labels):
                samples = self.samples_per_label[label]
                X.extend(samples)
                y.extend([label_idx] * len(samples))
            
            X = np.array(X)
            y = np.array(y)
            
            # Normalize training data
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = (X - mean) / std
            
            # Always use multiclass classification
            print(f"Training multiclass model with labels: {self.labels}")
            output_units = len(self.labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)  # Added stratify
            
            # Create and compile model with improved architecture
            self.model = keras.Sequential([
                # Input normalization layer
                layers.Lambda(lambda x: (x - mean) / std, input_shape=(X.shape[1],)),
                
                # First block
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # Second block
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # Third block
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                # Output layer
                layers.Dense(output_units, activation='softmax')
            ])
            
            # Compile with improved optimizer settings
            self.model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=0.001,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07
                ),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Add early stopping callback with better parameters
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Changed to monitor accuracy
                patience=30,
                restore_best_weights=True,
                min_delta=0.01  # 1% improvement threshold
            )
            
            # Add learning rate reduction with better parameters
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',  # Changed to monitor accuracy
                factor=0.5,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
            
            # Train with improved parameters
            history = self.model.fit(
                X_train, y_train,
                epochs=150,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate
            test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
            print(f'\nTest accuracy: {test_acc:.4f}')
            
            # Save model and data
            self._save_model()
            self._save_history(history.history)
            
        except Exception as e:
            print(f"Error during training: {e}")
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
            labels_path = os.path.join('models', f'{self.mode}_labels.json')
            with open(labels_path, 'w') as f:
                json.dump(self.labels, f)
            print(f"Labels saved to: {labels_path}")
            
            # Save training data with mode prefix
            data_path = os.path.join('models', f'{self.mode}_training_data.npz')
            np.savez(data_path, 
                     data=np.array([sample for samples in self.samples_per_label.values() for sample in samples]), 
                     labels=np.array([label_idx for label_idx, samples in enumerate(self.samples_per_label.values()) for _ in samples]))
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
    parser = argparse.ArgumentParser(description='RTTSSL Model Trainer')
    parser.add_argument('--mode', choices=['gesture', 'letter', 'word', 'face'],
                      help='Training mode (gesture, letter, word, face)')
    parser.add_argument('--type', choices=['realtime', 'dataset'],
                      help='Training type (realtime, dataset)')
    
    args = parser.parse_args()
    
    # Convert command line args to trainer parameters
    mode = args.mode
    training_type = args.type
    
    trainer = ModelTrainer(mode=mode, training_type=training_type)
    trainer.train()
