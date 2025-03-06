import time
import cv2
import numpy as np

class FPSCounter:
    def __init__(self):
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        self.smoothing_factor = 0.9 
        
    def update(self):
        # calculo FPS
        self.curr_frame_time = time.time()
        current_fps = 1 / (self.curr_frame_time - self.prev_frame_time)
        
        # smoothing dos FPS com uma média móvel exponencial
        self.fps = (self.smoothing_factor * self.fps + 
                   (1 - self.smoothing_factor) * current_fps)
        
        self.prev_frame_time = self.curr_frame_time
        
    def draw(self, frame):
        fps_text = f"FPS: {int(self.fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame
        
    def get_fps(self):
        return max(1, int(self.fps))  # garante o mínimo de 1 FPS para o gravador de vídeo
