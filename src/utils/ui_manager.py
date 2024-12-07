"""
UI Manager Module
Provides enhanced UI elements and visual effects for the application.
"""

import cv2
import numpy as np
from datetime import datetime

class UIManager:
    def __init__(self):
        self.overlay_alpha = 0.7
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'primary': (0, 255, 0),    # Green
            'secondary': (255, 191, 0), # Deep Sky Blue
            'warning': (0, 69, 255),    # Orange
            'error': (0, 0, 255),       # Red
            'text': (255, 255, 255),    # White
            'background': (0, 0, 0)     # Black
        }
        
    def create_overlay(self, frame):
        """Create a semi-transparent overlay"""
        overlay = frame.copy()
        return overlay
        
    def add_status_bar(self, frame, mode, fps=None, is_recording=False, face_mode=None):
        """Add a professional status bar with mode, FPS, and status indicators"""
        height, width = frame.shape[:2]
        
        # Status bar background
        cv2.rectangle(frame, (0, height-40), (width, height), 
                     self.colors['background'], -1)
        
        # Mode indicator
        mode_text = f"Mode: {mode.upper()}"
        cv2.putText(frame, mode_text, (10, height-15), self.font,
                   0.6, self.colors['secondary'], 2)
        
        # Face mode indicator (if in face mode)
        if mode == "face" and face_mode:
            face_text = f"Face Mode: {face_mode}"
            face_size = cv2.getTextSize(face_text, self.font, 0.6, 2)[0]
            cv2.putText(frame, face_text, (180, height-15), self.font,
                       0.6, self.colors['secondary'], 2)
        
        # Recording indicator
        if is_recording:
            rec_text = "Recording"
            rec_size = cv2.getTextSize(rec_text, self.font, 0.6, 2)[0]
            cv2.putText(frame, rec_text, (width-rec_size[0]-150, height-15),
                       self.font, 0.6, self.colors['warning'], 2)
        
        # FPS counter
        if fps is not None:
            fps_text = f"FPS: {int(fps)}"
            fps_size = cv2.getTextSize(fps_text, self.font, 0.6, 2)[0]
            cv2.putText(frame, fps_text, (width-fps_size[0]-10, height-15),
                       self.font, 0.6, self.colors['primary'], 2)
        
        # Current time
        time_text = datetime.now().strftime("%H:%M:%S")
        time_size = cv2.getTextSize(time_text, self.font, 0.6, 2)[0]
        cv2.putText(frame, time_text, (width//2-time_size[0]//2, height-15),
                   self.font, 0.6, self.colors['text'], 2)
                   
        return frame
        
    def add_notification(self, frame, text, type='info'):
        """Add a temporary notification message"""
        height, width = frame.shape[:2]
        color = self.colors.get(type, self.colors['primary'])
        
        # Background box
        text_size = cv2.getTextSize(text, self.font, 0.7, 2)[0]
        box_width = text_size[0] + 20
        box_height = text_size[1] + 20
        
        x = width // 2 - box_width // 2
        y = 50
        
        # Draw background box with animation
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height),
                     self.colors['background'], -1)
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height),
                     color, 2)
        
        # Draw text
        cv2.putText(frame, text, (x + 10, y + box_height - 10),
                   self.font, 0.7, color, 2)
                   
        return frame
        
    def draw_detection_box(self, frame, bbox, label=None, confidence=None):
        """Draw a detection box with label and confidence"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw box with gradient effect
        gradient = np.linspace(0, 1, 10)
        for i, alpha in enumerate(gradient):
            color = tuple(map(int, np.array(self.colors['primary']) * alpha))
            cv2.rectangle(frame, (x1-i, y1-i), (x2+i, y2+i), color, 1)
            
        # Add label and confidence
        if label or confidence:
            text = f"{label} {confidence*100:.1f}%" if confidence else label
            cv2.rectangle(frame, (x1, y1-25), (x1 + len(text)*12, y1),
                         self.colors['background'], -1)
            cv2.putText(frame, text, (x1, y1-7), self.font,
                       0.6, self.colors['primary'], 2)
                       
        return frame
        
    def draw_landmarks(self, frame, landmarks, connections=None, color_base=None):
        """Draw landmarks with enhanced visualization"""
        height, width = frame.shape[:2]
        color_base = color_base or self.colors['primary']
        
        # Draw landmarks with depth effect
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            
            # Adjust point size based on z-depth
            radius = max(3, int(5 * (1 + z)))
            
            # Create gradient effect
            for r in range(radius, 0, -1):
                alpha = r / radius
                color = tuple(map(int, np.array(color_base) * alpha))
                cv2.circle(frame, (x, y), r, color, -1)
                
        # Draw connections
        if connections:
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                x1 = int(start_point.x * width)
                y1 = int(start_point.y * height)
                x2 = int(end_point.x * width)
                y2 = int(end_point.y * height)
                
                # Draw line with gradient effect
                cv2.line(frame, (x1, y1), (x2, y2),
                        self.colors['secondary'], 2)
                        
        return frame
        
    def create_progress_bar(self, frame, value, max_value, label="Progress"):
        """Create a stylish progress bar"""
        height, width = frame.shape[:2]
        bar_width = width - 100
        bar_height = 20
        x = 50
        y = height - 80
        
        # Background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                     self.colors['background'], -1)
        
        # Progress
        progress_width = int(bar_width * (value / max_value))
        cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height),
                     self.colors['primary'], -1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                     self.colors['text'], 1)
        
        # Label
        text = f"{label}: {int(value/max_value*100)}%"
        cv2.putText(frame, text, (x, y-10), self.font,
                   0.6, self.colors['text'], 2)
                   
        return frame
        
    def add_performance_overlay(self, frame, metrics):
        """Add performance metrics overlay"""
        height, width = frame.shape[:2]
        margin = 10
        line_height = 20
        
        # Background box
        metrics_height = 100 if 'gpu_usage' not in metrics else 140
        cv2.rectangle(frame, 
                     (width - 200 - margin, margin),
                     (width - margin, metrics_height + margin),
                     self.colors['background'], -1)
        
        # Add metrics text
        y = margin + line_height
        cv2.putText(frame, f"FPS: {int(metrics['fps'])}", 
                   (width - 190, y), self.font, 0.5, self.colors['text'], 1)
                   
        y += line_height
        cv2.putText(frame, f"CPU: {metrics['cpu_usage']:.1f}%",
                   (width - 190, y), self.font, 0.5, self.colors['text'], 1)
                   
        y += line_height
        cv2.putText(frame, f"MEM: {metrics['memory_usage']:.1f}%",
                   (width - 190, y), self.font, 0.5, self.colors['text'], 1)
                   
        if 'gpu_usage' in metrics:
            y += line_height
            cv2.putText(frame, f"GPU: {metrics['gpu_usage']:.1f}%",
                      (width - 190, y), self.font, 0.5, self.colors['text'], 1)
            
            y += line_height
            cv2.putText(frame, f"VRAM: {metrics['gpu_memory']:.1f}%",
                      (width - 190, y), self.font, 0.5, self.colors['text'], 1)
        
        return frame
