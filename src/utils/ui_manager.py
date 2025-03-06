import cv2
import numpy as np
from datetime import datetime

class UIManager:
    def __init__(self):
        self.overlay_alpha = 0.7
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'primary': (0, 255, 0),     # verde
            'secondary': (255, 191, 0), # azul celeste
            'warning': (0, 69, 255),    # laranja
            'error': (0, 0, 255),       # vermelho
            'text': (255, 255, 255),    # branco
            'background': (0, 0, 0)     # preto
        }
        
    def create_overlay(self, frame):
        overlay = frame.copy()
        return overlay
        
    def add_status_bar(self, frame, mode, fps=None, is_recording=False, face_mode=None):
        height, width = frame.shape[:2]
        
        # status bar background
        cv2.rectangle(frame, (0, height-40), (width, height), 
                     self.colors['background'], -1)
        
        # mode indicator
        mode_text = f"Mode: {mode.upper()}"
        cv2.putText(frame, mode_text, (10, height-15), self.font,
                   0.6, self.colors['secondary'], 2)
        
        # face mode indicator (in face mode only)
        if mode == "face" and face_mode:
            face_text = f"Face Mode: {face_mode}"
            face_size = cv2.getTextSize(face_text, self.font, 0.6, 2)[0]
            cv2.putText(frame, face_text, (180, height-15), self.font,
                       0.6, self.colors['secondary'], 2)
        
        # recording indicator
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
        
        # current time
        time_text = datetime.now().strftime("%H:%M:%S")
        time_size = cv2.getTextSize(time_text, self.font, 0.6, 2)[0]
        cv2.putText(frame, time_text, (width//2-time_size[0]//2, height-15),
                   self.font, 0.6, self.colors['text'], 2)
                   
        return frame
        
    def add_notification(self, frame, text, type='info'):
        height, width = frame.shape[:2]
        color = self.colors.get(type, self.colors['primary'])
        
        text_size = cv2.getTextSize(text, self.font, 0.7, 2)[0]
        box_width = text_size[0] + 20
        box_height = text_size[1] + 20
        
        x = width // 2 - box_width // 2
        y = 50
    
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height),
                     self.colors['background'], -1)
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height),
                     color, 2)
        
        cv2.putText(frame, text, (x + 10, y + box_height - 10),
                   self.font, 0.7, color, 2)
                   
        return frame
        
    def draw_detection_box(self, frame, bbox, label=None, confidence=None):
        x1, y1, x2, y2 = map(int, bbox)
        
        gradient = np.linspace(0, 1, 10)
        for i, alpha in enumerate(gradient):
            color = tuple(map(int, np.array(self.colors['primary']) * alpha))
            cv2.rectangle(frame, (x1-i, y1-i), (x2+i, y2+i), color, 1)
            
        if label or confidence:
            text = f"{label} {confidence*100:.1f}%" if confidence else label
            cv2.rectangle(frame, (x1, y1-25), (x1 + len(text)*12, y1),
                         self.colors['background'], -1)
            cv2.putText(frame, text, (x1, y1-7), self.font,
                       0.6, self.colors['primary'], 2)
                       
        return frame
        
    def draw_landmarks(self, frame, landmarks, connections=None, color_base=None):
        height, width = frame.shape[:2]
        color_base = color_base or self.colors['primary']
        
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            
            radius = max(3, int(5 * (1 + z)))
            
            for r in range(radius, 0, -1):
                alpha = r / radius
                color = tuple(map(int, np.array(color_base) * alpha))
                cv2.circle(frame, (x, y), r, color, -1)

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
                
                cv2.line(frame, (x1, y1), (x2, y2),
                        self.colors['secondary'], 2)
                        
        return frame
        
    def create_progress_bar(self, frame, value, max_value, label="Progress"):
        height, width = frame.shape[:2]
        bar_width = width - 100
        bar_height = 20
        x = 50
        y = height - 80
        
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                     self.colors['background'], -1)
        
        progress_width = int(bar_width * (value / max_value))
        cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height),
                     self.colors['primary'], -1)
        
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                     self.colors['text'], 1)
        
        text = f"{label}: {int(value/max_value*100)}%"
        cv2.putText(frame, text, (x, y-10), self.font,
                   0.6, self.colors['text'], 2)
                   
        return frame
        
    def add_performance_overlay(self, frame, metrics):
       
        height, width = frame.shape[:2]
        margin = 10
        line_height = 20
        
        num_lines = 8 
        if 'gpu_usage' in metrics:
            num_lines += 6  
            
        metrics_height = margin + (num_lines * line_height) + margin
        
        cv2.rectangle(frame, 
                     (width - 300 - margin, margin),
                     (width - margin, metrics_height),
                     self.colors['background'], -1)
        
        y = int(margin + line_height)  
        x = int(width - 290)  
        
        cv2.putText(frame, f"FPS: {int(metrics['fps'])} (Min: {int(metrics['min_fps'])}, Max: {int(metrics['max_fps'])})", 
                   (x, y), self.font, 0.5, self.colors['primary'], 1)
        
        y = int(y + line_height)
        cv2.putText(frame, f"Frame Time: {metrics['avg_frame_time']:.1f}ms", 
                   (x, y), self.font, 0.5, self.colors['text'], 1)
        
        y = int(y + line_height * 1.5)
        cv2.putText(frame, f"CPU Usage: {metrics['cpu_usage']:.1f}% ({metrics['cpu_cores']} cores)", 
                   (x, y), self.font, 0.5, self.colors['secondary'], 1)
        
        y = int(y + line_height)
        cv2.putText(frame, f"CPU Frequency: {metrics['cpu_freq']:.0f}MHz",
                   (x, y), self.font, 0.5, self.colors['text'], 1)
        
        y = int(y + line_height * 1.5)
        cv2.putText(frame, f"RAM Usage: {metrics['memory_usage']:.1f}%",
                   (x, y), self.font, 0.5, self.colors['secondary'], 1)
        
        y = int(y + line_height)
        cv2.putText(frame, f"RAM Available: {metrics['memory_available']:.1f}GB / {metrics['memory_total']:.1f}GB",
                   (x, y), self.font, 0.5, self.colors['text'], 1)
        
        if 'gpu_usage' in metrics:
            y = int(y + line_height * 1.5)
            cv2.putText(frame, f"GPU Usage: {metrics['gpu_usage']:.1f}%",
                      (x, y), self.font, 0.5, self.colors['secondary'], 1)
            
            y = int(y + line_height)
            cv2.putText(frame, f"GPU Memory: {metrics['gpu_memory_used']:.0f}MB / {metrics['gpu_memory_total']:.0f}MB",
                      (x, y), self.font, 0.5, self.colors['text'], 1)
            
            y = int(y + line_height)
            cv2.putText(frame, f"GPU Temperature: {metrics['gpu_temp']}°C",
                      (x, y), self.font, 0.5, self.colors['text'], 1)
            
            y = int(y + line_height)
            cv2.putText(frame, f"GPU Power: {metrics['gpu_power']:.1f}W",
                      (x, y), self.font, 0.5, self.colors['text'], 1)
        
        y = int(y + line_height * 1.5)
        cv2.putText(frame, f"Total Frames: {metrics['total_frames']}",
                   (x, y), self.font, 0.5, self.colors['text'], 1)
        
        return frame
