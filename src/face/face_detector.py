"""
Face Detector Module
Provides functionality for face detection, recognition, and tracking
using OpenCV and MediaPipe Face Mesh.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load OpenCV's face detection cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Face recognition settings
        self.known_faces = []
        self.face_labels = []
        self.label_names = {}
        self.next_label = 0
        self.mode = "mesh"  # "mesh", "iris", or "recognition"
        self.model_file = os.path.join('models', 'face_recognizer.yml')  # Changed to .yml for OpenCV
        self.labels_file = os.path.join('models', 'face_labels.json')
        self.training_data_file = os.path.join('models', 'face_training_data.npz')
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load existing model and names
        self.load_recognizer()
        
    def load_recognizer(self):
        """Load the face recognizer model and names"""
        try:
            # Initialize face recognizer
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Load labels
            if os.path.exists(self.labels_file):
                with open(self.labels_file, 'r') as f:
                    names_list = json.load(f)
                    # Convert list to dictionary with indices as keys
                    self.label_names = {str(i): name for i, name in enumerate(names_list)}
                print(f"Loaded {len(self.label_names)} face labels: {self.label_names}")
            else:
                print("No face labels found")
                self.label_names = {}
            
            # Load training data
            if os.path.exists(self.training_data_file):
                data = np.load(self.training_data_file)
                print(f"Available keys in training data: {data.files}")
                
                if 'data' in data.files:
                    face_data = data['data']
                    face_labels = data['labels']
                    print(f"Loaded {len(face_data)} face samples using 'data' key")
                    print(f"Face data shape: {face_data.shape}")
                    print(f"Labels shape: {face_labels.shape}")
                    
                    # Ensure data is in the correct format (grayscale images)
                    self.known_faces = []
                    self.face_labels = []
                    for i in range(len(face_data)):
                        face = face_data[i]
                        if len(face.shape) == 3:  # If RGB, convert to grayscale
                            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                        # Ensure face is 100x100
                        face = cv2.resize(face, (100, 100))
                        self.known_faces.append(face)
                        self.face_labels.append(face_labels[i])
                    
                    # Convert to numpy arrays
                    self.known_faces = np.array(self.known_faces)
                    self.face_labels = np.array(self.face_labels, dtype=np.int32)
                    
                    print(f"Processed {len(self.known_faces)} faces for training")
                    print(f"Final data shape: {self.known_faces.shape}")
                    print(f"Final labels shape: {self.face_labels.shape}")
                    
                    # Train model with processed data
                    self.face_recognizer.train(self.known_faces, self.face_labels)
                    print("Successfully trained model with loaded faces")
                else:
                    print("No face data found in training file")
                    self.known_faces = []
                    self.face_labels = []
                    self.face_recognizer.train([np.zeros((100, 100), dtype=np.uint8)], np.array([0]))
            else:
                print("No training data found")
                self.known_faces = []
                self.face_labels = []
                self.face_recognizer.train([np.zeros((100, 100), dtype=np.uint8)], np.array([0]))
                
        except Exception as e:
            print(f"Error loading recognizer: {e}")
            print("Initializing with empty model")
            self.label_names = {}
            self.known_faces = []
            self.face_labels = []
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_recognizer.train([np.zeros((100, 100), dtype=np.uint8)], np.array([0]))
            
    def save_recognizer(self):
        """Save the face recognizer model and names"""
        try:
            # Save labels
            with open(self.labels_file, 'w') as f:
                json.dump(self.label_names, f, indent=4)
            print(f"Saved {len(self.label_names)} face labels")
            
            # Save training data
            if self.known_faces:
                np.savez(self.training_data_file, 
                        data=np.array(self.known_faces),
                        labels=np.array(self.face_labels))
                print(f"Saved {len(self.known_faces)} face samples")
            
            # Save model
            self.face_recognizer.write(self.model_file)
            print("Saved face recognition model")
            
        except Exception as e:
            print(f"Error saving recognizer: {e}")
            
    def add_face(self, frame, name):
        """Add a new face to the recognizer"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                print("No face detected in frame")
                return False
                
            if len(faces) > 1:
                print("Multiple faces detected. Please ensure only one face is visible.")
                return False
                
            # Get the face region
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to a standard size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Add to training data
            label = str(self.next_label)  # Convert to string
            self.known_faces.append(face_roi)
            self.face_labels.append(int(label))  # Convert back to int for training
            self.label_names[label] = name  # Use label_names instead of face_names
            self.next_label += 1
            
            # Save face image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join('faces', filename)
            cv2.imwrite(filepath, face_roi)
            
            # Retrain recognizer
            self.face_recognizer.train(self.known_faces, np.array(self.face_labels))
            self.save_recognizer()
            
            print(f"Added new face: {name}")
            return True
            
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
            
    def process_frame(self, frame):
        """Process frame and detect/recognize faces based on current mode"""
        if self.mode == "mesh":
            return self._process_mesh(frame)
        elif self.mode == "iris":
            return self._process_iris(frame)
        else:
            return self._process_recognition(frame)
            
    def _process_mesh(self, frame):
        """Process frame with face mesh detection"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._draw_face_mesh(frame, face_landmarks)
                
        return frame

    def _process_iris(self, frame):
        """Process frame with iris detection"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            height, width, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                # Left eye iris landmarks (indices 468-472)
                left_iris = []
                for idx in range(468, 473):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    left_iris.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 69, 255), -1)  # Orange points
                
                # Right eye iris landmarks (indices 473-477)
                right_iris = []
                for idx in range(473, 478):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    right_iris.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 69, 255), -1)  # Orange points
                
                # Draw iris circles
                if left_iris:
                    center = np.mean(left_iris, axis=0).astype(int)
                    radius = int(np.mean([np.linalg.norm(np.array(p) - center) for p in left_iris]))
                    cv2.circle(frame, tuple(center), radius, (0, 255, 0), 1)  # Green circle
                    cv2.circle(frame, tuple(center), 2, (0, 69, 255), -1)  # Orange center
                
                if right_iris:
                    center = np.mean(right_iris, axis=0).astype(int)
                    radius = int(np.mean([np.linalg.norm(np.array(p) - center) for p in right_iris]))
                    cv2.circle(frame, tuple(center), radius, (0, 255, 0), 1)  # Green circle
                    cv2.circle(frame, tuple(center), 2, (0, 69, 255), -1)  # Orange center
        
        return frame
        
    def _process_recognition(self, frame):
        """Process frame with face recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each face
        for (x, y, w, h) in faces:
            # Get face region
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            
            try:
                # Predict face
                label, distance = self.face_recognizer.predict(face_roi)
                # Convert distance to confidence (0-100%)
                # LBPH típico tem distâncias entre 0-100, onde menor é melhor
                confidence = max(0, min(100, 100 * (1 - distance/100)))
                
                # Get name from label
                label_str = str(label)
                name = self.label_names.get(label_str, "Unknown")
                
                # Ajustando o limiar de confiança
                if confidence < 30:  # Usando 30% como limiar
                    name = "Unknown"
                    print(f"Face detected but confidence too low: {confidence:.1f}% (distance: {distance:.1f})")
                else:
                    name = f"{name} ({confidence:.1f}%)"
                    print(f"Face recognized: {name} (distance: {distance:.1f})")
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw name background
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, 
                            (x - 2, y - text_size[1] - 10),
                            (x + text_size[0] + 2, y),
                            (0, 0, 0), -1)
                
                # Draw name
                cv2.putText(frame, name,
                           (x, y - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                           
            except Exception as e:
                print(f"Error in face recognition: {e}")
                print(f"Label: {label if 'label' in locals() else 'unknown'}")
                print(f"Label names: {self.label_names}")
        
        return frame
        
    def _draw_face_mesh(self, frame, landmarks):
        """Draw face mesh landmarks on the frame"""
        height, width, _ = frame.shape
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
    def get_mode(self):
        """Get current face detection mode"""
        return self.mode
        
    def toggle_mode(self):
        """Toggle between face modes"""
        if self.mode == "mesh":
            self.mode = "iris"
        elif self.mode == "iris":
            self.mode = "recognition"
            # Load face recognizer model if it exists
            if os.path.exists(self.model_file):
                try:
                    self.face_recognizer.read(self.model_file)
                    print("Loaded face recognition model")
                except Exception as e:
                    print(f"Error loading face model: {e}")
        else:
            self.mode = "mesh"
