"""
Face Detector Module
Provides functionality for face detection, mesh visualization, and iris tracking
using MediaPipe Face Detection and Face Mesh.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.mode = "mesh"  # "mesh" or "iris"
        self.face_names = {}  # Dictionary to store face names
        self.names_file = os.path.join('data', 'face_names.json')
        self._load_face_names()  # Carregar nomes salvos
        
    def _load_face_names(self):
        """Load saved face names from file"""
        try:
            os.makedirs('data', exist_ok=True)
            if os.path.exists(self.names_file):
                with open(self.names_file, 'r') as f:
                    self.face_names = {int(k): v for k, v in json.load(f).items()}
        except Exception as e:
            print(f"Error loading face names: {e}")
            self.face_names = {}
            
    def _save_face_names(self):
        """Save face names to file"""
        try:
            os.makedirs('data', exist_ok=True)
            with open(self.names_file, 'w') as f:
                json.dump(self.face_names, f, indent=4)
        except Exception as e:
            print(f"Error saving face names: {e}")
            
    def process_frame(self, frame):
        """Process frame and detect faces based on current mode"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.mode == "mesh":
            return self._process_mesh(frame, frame_rgb)
        else:
            return self._process_iris(frame, frame_rgb)
            
    def _process_mesh(self, frame, frame_rgb):
        """Process frame with face mesh detection"""
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                self._draw_face_mesh(frame, face_landmarks)
                # Draw face name if exists
                if idx in self.face_names:
                    self._draw_face_name(frame, face_landmarks, self.face_names[idx])
                
        return frame
        
    def _process_iris(self, frame, frame_rgb):
        """Process frame with iris detection"""
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                self._draw_iris(frame, face_landmarks)
                # Draw face name if exists
                if idx in self.face_names:
                    self._draw_face_name(frame, face_landmarks, self.face_names[idx])
                
        return frame
        
    def _draw_face_mesh(self, frame, landmarks):
        """Draw face mesh landmarks on the frame"""
        height, width, _ = frame.shape
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
    def _draw_iris(self, frame, landmarks):
        """Draw iris landmarks on the frame"""
        height, width, _ = frame.shape
        
        # Left eye iris landmarks (indices 468-472)
        left_iris = []
        for idx in range(468, 473):  # 5 pontos para a íris esquerda
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            left_iris.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 69, 255), -1)  # Pontos em laranja
            
        # Right eye iris landmarks (indices 473-477)
        right_iris = []
        for idx in range(473, 478):  # 5 pontos para a íris direita
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            right_iris.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 69, 255), -1)  # Pontos em laranja
            
        # Draw iris circles
        if left_iris:
            center = np.mean(left_iris, axis=0).astype(int)
            radius = int(np.mean([np.linalg.norm(np.array(p) - center) for p in left_iris]))
            cv2.circle(frame, tuple(center), radius, (0, 255, 0), 1)  # Círculo verde
            cv2.circle(frame, tuple(center), 2, (0, 69, 255), -1)  # Centro em laranja
            
        if right_iris:
            center = np.mean(right_iris, axis=0).astype(int)
            radius = int(np.mean([np.linalg.norm(np.array(p) - center) for p in right_iris]))
            cv2.circle(frame, tuple(center), radius, (0, 255, 0), 1)  # Círculo verde
            cv2.circle(frame, tuple(center), 2, (0, 69, 255), -1)  # Centro em laranja
            
    def _draw_face_name(self, frame, landmarks, name):
        """Draw name above the detected face"""
        height, width, _ = frame.shape
        
        # Get the top of the face
        min_y = float('inf')
        for landmark in landmarks.landmark:
            y = int(landmark.y * height)
            if y < min_y:
                min_y = y
                x = int(landmark.x * width)
        
        # Draw name background
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, 
                     (x - text_size[0]//2 - 10, min_y - 40),
                     (x + text_size[0]//2 + 10, min_y - 10),
                     (0, 0, 0), -1)
        
        # Draw name
        cv2.putText(frame, name,
                   (x - text_size[0]//2, min_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    def get_mode(self):
        """Get current face detection mode"""
        return self.mode
        
    def toggle_mode(self):
        """Toggle between face mesh and iris detection modes"""
        self.mode = "iris" if self.mode == "mesh" else "mesh"
        
    def add_face_name(self, face_id, name):
        """Add or update name for a detected face"""
        self.face_names[face_id] = name
        self._save_face_names()
