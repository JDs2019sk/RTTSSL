"""
Face Detector Module
Provides functionality for face detection, mesh visualization, and iris tracking
using MediaPipe Face Detection and Face Mesh.
"""

import cv2
import mediapipe as mp
import numpy as np

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
            for face_landmarks in results.multi_face_landmarks:
                self._draw_face_mesh(frame, face_landmarks)
                
        return frame
        
    def _process_iris(self, frame, frame_rgb):
        """Process frame with iris detection"""
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._draw_iris(frame, face_landmarks)
                
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
        
        # Left eye iris landmarks (468-478)
        left_iris = []
        for idx in range(468, 478):
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            left_iris.append((x, y))
            
        # Right eye iris landmarks (473-477)
        right_iris = []
        for idx in range(473, 478):
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            right_iris.append((x, y))
            
        # Draw iris circles
        if left_iris:
            center = np.mean(left_iris, axis=0).astype(int)
            cv2.circle(frame, tuple(center), 3, (0, 255, 0), -1)
            
        if right_iris:
            center = np.mean(right_iris, axis=0).astype(int)
            cv2.circle(frame, tuple(center), 3, (0, 255, 0), -1)
            
    def toggle_mode(self):
        """Toggle between mesh and iris detection modes"""
        self.mode = "iris" if self.mode == "mesh" else "mesh"
        
    def add_face_name(self, face_id, name):
        """Add or update name for a detected face"""
        self.face_names[face_id] = name
