"""
RTTSSL
Programa principal.
"""

import cv2
import keyboard
import yaml
import os
import threading
import time
from src.gesture.gesture_recognizer import GestureRecognizer
from src.face.face_detector import FaceDetector
from src.mouse.hand_controller import HandController
from src.utils.fps_counter import FPSCounter
from src.utils.help_menu import HelpMenu
from src.utils.ui_manager import UIManager
from src.utils.performance_monitor import PerformanceMonitor

class RTTSSL:
    def __init__(self):
        self.config = self._load_config()
        self.gesture_recognizer = GestureRecognizer()
        self.face_detector = FaceDetector()
        self.hand_controller = HandController()
        self.fps_counter = FPSCounter()
        self.help_menu = HelpMenu()
        self.ui_manager = UIManager()
        self.performance_monitor = PerformanceMonitor()
        self.show_fps = False
        self.show_help = False
        self.show_performance = False
        self.active_mode = "gesture"
        self.running = True
        self.recording = False
        self.last_notification = None
        self.notification_time = 0
        self.key_states = {}
        self.key_cooldown = 0.2  
        self.last_key_press = {}
        self.window_name = "RTTSSL"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.current_translation = ""
        self.translation_time = 0
        self.translation_duration = 3.0  
        
    def _load_config(self):
        config_path = os.path.join('config', 'configs.yaml')
        default_config = {
            'keybinds': {
                'letter_mode': '1',
                'word_mode': '2',
                'gesture_mode': '3',
                'mouse_control': 'm',
                'face_detection': 'f',
                'toggle_detection_mode': 'e',
                'toggle_fps': 'tab',
                'toggle_performance': 'p',
                'toggle_recording': 'r',
                'help_menu': 'h'
            },
            'modes': {
                'letter': {'enabled': True, 'description': 'Translate sign language letters'},
                'word': {'enabled': True, 'description': 'Translate sign language words'},
                'gesture': {'enabled': True, 'description': 'Recognize and translate gestures'},
                'mouse': {'enabled': True, 'description': 'Control mouse with hand gestures'},
                'face': {
                    'enabled': True,
                    'submodes': ['mesh', 'iris'],
                    'description': 'Face detection and recognition'
                }
            },
            'performance': {
                'target_fps': 60,
                'enable_threading': True,
                'enable_gpu': True,
                'optimize_resolution': True,
                'notification_duration': 3.5
            }
        }

        try:
            os.makedirs('config', exist_ok=True)
            
            if not os.path.exists(config_path):
                with open(config_path, 'w') as file:
                    yaml.dump(default_config, file, default_flow_style=False, sort_keys=False)
                print("Created default configuration file")
                return default_config
                
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            if not all(key in config for key in default_config.keys()):
                print("Warning: Missing keys in config file. Using defaults for missing keys.")
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                with open(config_path, 'w') as file:
                    yaml.dump(config, file, default_flow_style=False, sort_keys=False)
                    
            return config
                
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
            return default_config
            
    def _can_press_key(self, key):
        current_time = time.time()
        if key not in self.last_key_press:
            self.last_key_press[key] = 0
        if current_time - self.last_key_press[key] >= self.key_cooldown:
            self.last_key_press[key] = current_time
            return True
        return False
            
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
        
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                self.performance_monitor.start_frame()
                
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                display_frame = frame.copy()
                
                if self.active_mode in ["gesture", "letter", "word"]:
                    frame, translation = self.gesture_recognizer.process_frame(frame)
                    if translation:
                        self._update_translation(translation)
                elif self.active_mode == "face":
                    frame = self.face_detector.process_frame(frame)
                elif self.active_mode == "mouse":
                    frame = self.hand_controller.process_frame(frame)
                    
                self.fps_counter.update()
                metrics = self.performance_monitor.get_metrics()
                
                frame = self.ui_manager.add_status_bar(
                    frame, self.active_mode,
                    fps=metrics['fps'] if self.show_fps else None,
                    is_recording=self.recording,
                    face_mode="Mesh" if self.active_mode == "face" and self.face_detector.mode == "mesh" else
                             "Iris" if self.active_mode == "face" and self.face_detector.mode == "iris" else None
                )
                
                if self.current_translation:
                    frame = self._draw_translation_box(frame)
                
                if self.show_performance:
                    frame = self._draw_performance_metrics(frame, metrics)
                    
                if self.show_help:
                    frame = self.help_menu.draw(frame)
                    
                if self.last_notification:
                    self._update_notification(frame)
                
                if self.recording:
                    self._record_frame(frame)
                
                cv2.imshow(self.window_name, frame)
                
                keybinds = self.config['keybinds']
                modes = self.config['modes']
                
                if keyboard.is_pressed(keybinds['letter_mode']) and self._can_press_key('letter_mode'):
                    if modes['letter']['enabled']:
                        self.active_mode = "letter"
                        self.gesture_recognizer.set_mode("letter")
                        self._show_notification("Letter Translation Mode", 'info')
                    
                elif keyboard.is_pressed(keybinds['word_mode']) and self._can_press_key('word_mode'):
                    if modes['word']['enabled']:
                        self.active_mode = "word"
                        self.gesture_recognizer.set_mode("word")
                        self._show_notification("Word Translation Mode", 'info')
                    
                elif keyboard.is_pressed(keybinds['gesture_mode']) and self._can_press_key('gesture_mode'):
                    if modes['gesture']['enabled']:
                        self.active_mode = "gesture"
                        self.gesture_recognizer.set_mode("gesture")
                        self._show_notification("Gesture Recognition Mode", 'info')
                    
                elif keyboard.is_pressed(keybinds['mouse_control']) and self._can_press_key('mouse_control'):
                    if modes['mouse']['enabled']:
                        self.active_mode = "mouse" if self.active_mode != "mouse" else "gesture"
                        self._show_notification(
                            "Mouse Control Enabled" if self.active_mode == "mouse" else "Mouse Control Disabled",
                            'info'
                        )
                    
                elif keyboard.is_pressed(keybinds['face_detection']) and self._can_press_key('face_detection'):
                    if modes['face']['enabled']:
                        self.active_mode = "face" if self.active_mode != "face" else "gesture"
                        self._show_notification(
                            "Face Detection Enabled" if self.active_mode == "face" else "Face Detection Disabled",
                            'info'
                        )
                    
                elif keyboard.is_pressed(keybinds['toggle_detection_mode']) and self._can_press_key('toggle_detection'):
                    if self.active_mode == "face" and modes['face']['enabled']:
                        self.face_detector.toggle_mode()
                        current_mode = self.face_detector.get_mode()
                        self._show_notification(f"Face Detection Mode: {current_mode.capitalize()}", 'info')
                        
                elif keyboard.is_pressed(keybinds['toggle_performance']) and self._can_press_key('toggle_performance'):
                    self.show_performance = not self.show_performance
                    
                elif keyboard.is_pressed(keybinds['toggle_fps']) and self._can_press_key('toggle_fps'):
                    self.show_fps = not self.show_fps
                    
                elif keyboard.is_pressed(keybinds['help_menu']) and self._can_press_key('help_menu'):
                    self.show_help = not self.show_help
                    
                elif keyboard.is_pressed(keybinds['toggle_recording']) and self._can_press_key('toggle_recording'):
                    self.recording = not self.recording
                    if not self.recording and hasattr(self, 'video_writer'):
                        self.video_writer.release()
                        delattr(self, 'video_writer')
                    self._show_notification(
                        "Recording Started" if self.recording else "Recording Stopped",
                        'warning' if self.recording else 'info'
                    )
                    
                if cv2.waitKey(1) & 0xFF == 27: 
                    break
                    
                self.performance_monitor.end_frame()
                
        except KeyboardInterrupt:
            print("\nClosing Programme.")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            print("\nCleaning...")
            if hasattr(self, 'video_writer'):
                self.video_writer.release()
            if self.face_detector and hasattr(self.face_detector, '_save_face_names'):
                self.face_detector._save_face_names()
            self.performance_monitor.cleanup()
            cap.release()
            cv2.destroyAllWindows()
            print("Programme closed.")

    def _update_translation(self, translation):
        self.current_translation = translation
        self.translation_time = time.time()
        
    def _draw_translation_box(self, frame):
        if not self.current_translation:
            return frame
            
        current_time = time.time()
        if current_time - self.translation_time > self.translation_duration:
            self.current_translation = ""
            return frame
            
        height, width = frame.shape[:2]
        
        text_size = cv2.getTextSize(self.current_translation, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        box_width = text_size[0] + 40
        box_height = text_size[1] + 40
        
        x = (width - box_width) // 2
        y = height - box_height - 100
        
        overlay = frame.copy()
        
        cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height),
                     (0, 0, 0), -1)
        
        text_x = x + 20
        text_y = y + box_height - 20
        cv2.putText(overlay, self.current_translation,
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                   
        alpha = min(1.0, (self.translation_duration - (current_time - self.translation_time)) / 0.5)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
        
    def _handle_keyboard(self):
        keybinds = self.config['keybinds']
        modes = self.config['modes']
        
        if keyboard.is_pressed(keybinds['letter_mode']) and self._can_press_key('letter_mode'):
            if modes['letter']['enabled']:
                self.active_mode = "letter"
                self.gesture_recognizer.set_mode("letter")
                self._show_notification("Letter Translation Mode", 'info')
            
        elif keyboard.is_pressed(keybinds['word_mode']) and self._can_press_key('word_mode'):
            if modes['word']['enabled']:
                self.active_mode = "word"
                self.gesture_recognizer.set_mode("word")
                self._show_notification("Word Translation Mode", 'info')
            
        elif keyboard.is_pressed(keybinds['gesture_mode']) and self._can_press_key('gesture_mode'):
            if modes['gesture']['enabled']:
                self.active_mode = "gesture"
                self.gesture_recognizer.set_mode("gesture")
                self._show_notification("Gesture Recognition Mode", 'info')
            
        elif keyboard.is_pressed(keybinds['mouse_control']) and self._can_press_key('mouse_control'):
            if modes['mouse']['enabled']:
                self.active_mode = "mouse" if self.active_mode != "mouse" else "gesture"
                self._show_notification(
                    "Mouse Control Enabled" if self.active_mode == "mouse" else "Mouse Control Disabled",
                    'info'
                )
            
        elif keyboard.is_pressed(keybinds['face_detection']) and self._can_press_key('face_detection'):
            if modes['face']['enabled']:
                self.active_mode = "face" if self.active_mode != "face" else "gesture"
                self._show_notification(
                    "Face Detection Enabled" if self.active_mode == "face" else "Face Detection Disabled",
                    'info'
                )
            
        elif keyboard.is_pressed(keybinds['toggle_detection_mode']) and self._can_press_key('toggle_detection'):
            if self.active_mode == "face" and modes['face']['enabled']:
                self.face_detector.toggle_mode()
                current_mode = self.face_detector.get_mode()
                self._show_notification(f"Face Detection Mode: {current_mode.capitalize()}", 'info')
                
        elif keyboard.is_pressed(keybinds['toggle_performance']) and self._can_press_key('toggle_performance'):
            self.show_performance = not self.show_performance
            
        elif keyboard.is_pressed(keybinds['toggle_fps']) and self._can_press_key('toggle_fps'):
            self.show_fps = not self.show_fps
            
        elif keyboard.is_pressed(keybinds['help_menu']) and self._can_press_key('help_menu'):
            self.show_help = not self.show_help
            
        elif keyboard.is_pressed(keybinds['toggle_recording']) and self._can_press_key('toggle_recording'):
            self.recording = not self.recording
            if not self.recording and hasattr(self, 'video_writer'):
                self.video_writer.release()
                delattr(self, 'video_writer')
            self._show_notification(
                "Recording Started" if self.recording else "Recording Stopped",
                'warning' if self.recording else 'info'
            )
            
    def _show_notification(self, text, type='info'):
        self.last_notification = (text, type)
        self.notification_time = time.time()
        
    def _update_notification(self, frame):
        if self.last_notification:
            current_time = time.time()
            duration = current_time - self.notification_time
            
            if duration < 2.0: 
                text, type = self.last_notification
                frame = self.ui_manager.add_notification(frame, text, type)
            else:
                self.last_notification = None
                
    def _draw_performance_metrics(self, frame, metrics):
        return self.ui_manager.add_performance_overlay(frame, metrics)

    def _record_frame(self, frame):
        if not hasattr(self, 'video_writer'):
            os.makedirs('recordings', exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join('recordings', f'recording_{timestamp}.mp4')
            
            height, width = frame.shape[:2]
            fps = self.fps_counter.get_fps()
            if fps < 10:  
                fps = 30
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
        self.video_writer.write(frame)

if __name__ == "__main__":
    app = RTTSSL()
    app.run()
