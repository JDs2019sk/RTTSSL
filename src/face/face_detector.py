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
        
        # Initialize face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Face recognition settings
        self.known_faces = []
        self.face_labels = []
        self.label_names = {}
        self.next_label = 0
        self.mode = "mesh"  # "mesh", "iris", or "recognition"
        self.faces_dir = os.path.join('data', 'faces')
        self.model_file = os.path.join('data', 'face_recognizer.yml')
        self.names_file = os.path.join('data', 'face_names.json')
        
        # Create directories if they don't exist
        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        
        # Load existing model and names
        self.load_recognizer()
        
    def load_recognizer(self):
        """Load the face recognizer model and names"""
        try:
            # Load face names
            if os.path.exists(self.names_file):
                with open(self.names_file, 'r') as f:
                    self.label_names = json.load(f)
                    if self.label_names:
                        self.next_label = max(map(int, self.label_names.keys())) + 1
            else:
                self.label_names = {}
                self.next_label = 0
            
            # Load saved faces
            self.known_faces = []
            self.face_labels = []
            
            if os.path.exists(self.faces_dir):
                for filename in os.listdir(self.faces_dir):
                    if filename.endswith('.jpg'):
                        # Load face image
                        face_path = os.path.join(self.faces_dir, filename)
                        face_img = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
                        if face_img is not None:
                            # Get label from filename (format: name_timestamp.jpg)
                            name = filename.split('_')[0]
                            # Find label for this name
                            label = None
                            for lbl, n in self.label_names.items():
                                if n == name:
                                    label = int(lbl)
                                    break
                            if label is not None:
                                self.known_faces.append(face_img)
                                self.face_labels.append(label)
            
            # Train recognizer with loaded faces
            if self.known_faces:
                self.face_recognizer.train(self.known_faces, np.array(self.face_labels))
                print(f"Trained recognizer with {len(self.known_faces)} faces")
            else:
                # Train with empty data to initialize the model
                self.face_recognizer.train([np.zeros((100, 100), dtype=np.uint8)], np.array([0]))
                print("No faces found, initialized empty model")
                
        except Exception as e:
            print(f"Error loading recognizer: {e}")
            self.label_names = {}
            self.next_label = 0
            # Train with empty data to initialize the model
            self.face_recognizer.train([np.zeros((100, 100), dtype=np.uint8)], np.array([0]))
            
    def save_recognizer(self):
        """Save the face recognizer model and names"""
        try:
            # Save face names
            with open(self.names_file, 'w') as f:
                json.dump(self.label_names, f, indent=4)
            
            # Save recognizer model
            self.face_recognizer.write(self.model_file)
            print("Saved face recognizer model and names")
            
        except Exception as e:
            print(f"Error saving face recognizer: {e}")
            
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
            filepath = os.path.join(self.faces_dir, filename)
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
                label, confidence = self.face_recognizer.predict(face_roi)
                confidence = 100 - confidence  # Convert to percentage (higher is better)
                
                # Get name
                name = self.label_names.get(str(label), "Unknown")
                if confidence < 40:  # Confidence threshold
                    name = "Unknown"
                else:
                    name = f"{name} ({confidence:.1f}%)"
                
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
                print(f"Error recognizing face: {e}")
        
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
        else:
            self.mode = "mesh"
