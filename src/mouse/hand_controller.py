"""
Hand Controller Module
Provides functionality for controlling the mouse cursor using hand gestures
captured through the webcam.
"""

import cv2
import mediapipe as mp
import mouse
import numpy as np

class HandController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.screen_width, self.screen_height = self._get_screen_resolution()
        self.smoothing_factor = 0.5
        self.prev_x, self.prev_y = 0, 0
        
    def _get_screen_resolution(self):
        """Get the screen resolution for mouse mapping"""
        import tkinter as tk
        root = tk.Tk()
        return root.winfo_screenwidth(), root.winfo_screenheight()
        
    def process_frame(self, frame):
        """Process frame and control mouse based on hand position"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get index finger tip coordinates
            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * self.screen_width)
            y = int(index_tip.y * self.screen_height)
            
            # Apply smoothing
            x = int(self.smoothing_factor * x + (1 - self.smoothing_factor) * self.prev_x)
            y = int(self.smoothing_factor * y + (1 - self.smoothing_factor) * self.prev_y)
            
            # Move mouse
            mouse.move(x, y)
            
            # Check for click gesture (thumb tip close to index tip)
            thumb_tip = hand_landmarks.landmark[4]
            distance = np.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2
            )
            
            if distance < 0.05:  # Threshold for click
                mouse.click()
            
            self.prev_x, self.prev_y = x, y
            
            # Draw hand landmarks
            self._draw_landmarks(frame, hand_landmarks)
            
        return frame
        
    def _draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on the frame"""
        height, width, _ = frame.shape
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
