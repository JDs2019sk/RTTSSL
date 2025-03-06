"""
(MouseControl)
"""

import cv2
import mediapipe as mp
import mouse
import numpy as np
import time

class HandController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.screen_width, self.screen_height = self._get_screen_resolution()
        self.smoothing_factor = 0.7
        self.prev_x, self.prev_y = 0, 0
        self.click_cooldown = False
        self.last_click_time = 0
        self.click_cooldown_time = 0.3
        self.movement_multiplier = 1.5
        
        self.position_buffer_size = 3
        self.x_buffer = []
        self.y_buffer = []

    def _get_screen_resolution(self):
        """Obtém a resolução do ecrã para mapeamento do rato"""
        import tkinter as tk
        root = tk.Tk()
        return root.winfo_screenwidth(), root.winfo_screenheight()
        
    def _average_position(self, x, y):
        """Calcula a média móvel das posições para suavizar o movimento"""
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        if len(self.x_buffer) > self.position_buffer_size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)
            
        return (sum(self.x_buffer) / len(self.x_buffer),
                sum(self.y_buffer) / len(self.y_buffer))

    def process_frame(self, frame):
        """Processa frame e controla o rato baseado na posição da mão"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            
            index_base = hand_landmarks.landmark[5]
            middle_base = hand_landmarks.landmark[9]
            
            index_mid = hand_landmarks.landmark[7]
            middle_mid = hand_landmarks.landmark[11]
            
            index_raised = (index_tip.y < index_mid.y < index_base.y and 
                          abs(index_tip.x - index_base.x) < 0.1)
            middle_raised = (middle_tip.y < middle_mid.y < middle_base.y and 
                           abs(middle_tip.x - middle_base.x) < 0.1)
            
            center_x = self.screen_width / 2
            center_y = self.screen_height / 2
            
            x_offset = ((1.0 - index_tip.x) - 0.5) * self.movement_multiplier
            y_offset = (index_tip.y - 0.5) * self.movement_multiplier
            
            x = int(center_x + (x_offset * self.screen_width))
            y = int(center_y + (y_offset * self.screen_height))
            
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))
            
            x, y = self._average_position(x, y)
            
            x = int(self.smoothing_factor * x + (1 - self.smoothing_factor) * self.prev_x)
            y = int(self.smoothing_factor * y + (1 - self.smoothing_factor) * self.prev_y)
            
            if abs(x - self.prev_x) > 1 or abs(y - self.prev_y) > 1:
                mouse.move(int(x), int(y))
            
            if index_raised and middle_raised:
                if not self.click_cooldown:
                    mouse.click()
                    self.click_cooldown = True
                    self.last_click_time = time.time()
            else:
                if self.click_cooldown and time.time() - self.last_click_time > self.click_cooldown_time:
                    self.click_cooldown = False
            
            self.prev_x, self.prev_y = x, y
            
            self._draw_landmarks(frame, hand_landmarks)
            
            status = "| Click!" if index_raised and middle_raised else "* Move"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            
        return frame
        
    def _draw_landmarks(self, frame, landmarks):
        """Desenha os pontos de referência da mão com conexões"""
        height, width = frame.shape[:2]
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 17), (5, 9), (9, 13), (13, 17),
            (0, 5)
        ]
        
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
            
        for connection in connections:
            start_point = points[connection[0]]
            end_point = points[connection[1]]
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
            
        for point in points:
            cv2.circle(frame, point, 4, (0, 255, 0), -1)
            cv2.circle(frame, point, 5, (0, 255, 0), 1)
