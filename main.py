"""
RTTSSL (Real-Time Translation of Signs, Speech, and Letters)
Main program that integrates all components for gesture recognition,
face detection, and mouse control with enhanced visuals and performance.
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
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.gesture_recognizer = GestureRecognizer()
        self.face_detector = FaceDetector()
        self.hand_controller = HandController()
        self.fps_counter = FPSCounter()
        self.help_menu = HelpMenu()
        self.ui_manager = UIManager()
        self.performance_monitor = PerformanceMonitor()
        
        # State flags
        self.show_fps = False
        self.show_help = False
        self.show_performance = False
        self.active_mode = "gesture"
        self.running = True
        self.recording = False
        self.last_notification = None
        self.notification_time = 0
        
        # Key state tracking
        self.key_states = {}
        self.key_cooldown = 0.2  # seconds
        self.last_key_press = {}
        
        # Window settings
        self.window_name = "RTTSSL"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Translation box
        self.current_translation = ""
        self.translation_time = 0
        self.translation_duration = 3.0  # seconds
        
    def _load_config(self):
        """Load or create keybinds configuration"""
        config_path = os.path.join('config', 'keybinds.yaml')
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
                'target_fps': 30,
                'enable_threading': True,
                'enable_gpu': True,
                'optimize_resolution': True,
                'notification_duration': 3.0
            }
        }

        try:
            # Create config directory if it doesn't exist
            os.makedirs('config', exist_ok=True)
            
            # If file doesn't exist, create with default configuration
            if not os.path.exists(config_path):
                with open(config_path, 'w') as file:
                    yaml.dump(default_config, file, default_flow_style=False, sort_keys=False)
                print("Created default configuration file")
                return default_config
                
            # Load existing configuration
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Check if all required keys exist
            if not all(key in config for key in default_config.keys()):
                print("Warning: Missing keys in config file. Using defaults for missing keys.")
                # Update configuration with default values for missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                # Save updated configuration
                with open(config_path, 'w') as file:
                    yaml.dump(config, file, default_flow_style=False, sort_keys=False)
                    
            return config
                
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
            return default_config
            
    def _can_press_key(self, key):
        """Check if enough time has passed since last key press"""
        current_time = time.time()
        if key not in self.last_key_press:
            self.last_key_press[key] = 0
        if current_time - self.last_key_press[key] >= self.key_cooldown:
            self.last_key_press[key] = current_time
            return True
        return False
            
    def run(self):
        """Main program loop"""
        cap = cv2.VideoCapture(0)
        
        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Set window properties
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                self.performance_monitor.start_frame()
                
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Create a clean copy of the frame
                display_frame = frame.copy()
                
                # Process frame based on active mode
                if self.active_mode in ["gesture", "letter", "word"]:
                    frame, translation = self.gesture_recognizer.process_frame(frame)
                    if translation:
                        self._update_translation(translation)
                elif self.active_mode == "face":
                    frame = self.face_detector.process_frame(frame)
                elif self.active_mode == "mouse":
                    frame = self.hand_controller.process_frame(frame)
                    
                # Update and draw UI elements
                self.fps_counter.update()
                metrics = self.performance_monitor.get_metrics()
                
                # Add status bar
                frame = self.ui_manager.add_status_bar(
                    frame, self.active_mode,
                    fps=metrics['fps'] if self.show_fps else None,
                    is_recording=self.recording,
                    face_mode="Mesh" if self.active_mode == "face" and self.face_detector.mode == "mesh" else
                             "Iris" if self.active_mode == "face" and self.face_detector.mode == "iris" else None
                )
                
                # Show current translation
                if self.current_translation:
                    frame = self._draw_translation_box(frame)
                
                # Show performance metrics
                if self.show_performance:
                    self._draw_performance_metrics(frame, metrics)
                    
                # Show help menu
                if self.show_help:
                    frame = self.help_menu.draw(frame)
                    
                # Show notification if exists
                if self.last_notification:
                    self._update_notification(frame)
                    
                # Record if enabled
                if self.recording:
                    self._record_frame(frame)
                    
                # Display frame
                cv2.imshow(self.window_name, frame)
                
                # Handle keyboard events
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
                    
                # Mouse control
                elif keyboard.is_pressed(keybinds['mouse_control']) and self._can_press_key('mouse_control'):
                    if modes['mouse']['enabled']:
                        self.active_mode = "mouse" if self.active_mode != "mouse" else "gesture"
                        self._show_notification(
                            "Mouse Control Enabled" if self.active_mode == "mouse" else "Mouse Control Disabled",
                            'info'
                        )
                    
                # Face detection
                elif keyboard.is_pressed(keybinds['face_detection']) and self._can_press_key('face_detection'):
                    if modes['face']['enabled']:
                        self.active_mode = "face" if self.active_mode != "face" else "gesture"
                        self._show_notification(
                            "Face Detection Enabled" if self.active_mode == "face" else "Face Detection Disabled",
                            'info'
                        )
                    
                # Toggle detection mode (face mesh/iris)
                elif keyboard.is_pressed(keybinds['toggle_detection_mode']) and self._can_press_key('toggle_detection'):
                    if self.active_mode == "face" and modes['face']['enabled']:
                        self.face_detector.toggle_mode()
                        current_mode = self.face_detector.get_mode()
                        self._show_notification(f"Face Detection Mode: {current_mode.capitalize()}", 'info')
                        
                # Performance display
                elif keyboard.is_pressed(keybinds['toggle_performance']) and self._can_press_key('toggle_performance'):
                    self.show_performance = not self.show_performance
                    
                # FPS display
                elif keyboard.is_pressed(keybinds['toggle_fps']) and self._can_press_key('toggle_fps'):
                    self.show_fps = not self.show_fps
                    
                # Help menu
                elif keyboard.is_pressed(keybinds['help_menu']) and self._can_press_key('help_menu'):
                    self.show_help = not self.show_help
                    
                # Recording
                elif keyboard.is_pressed(keybinds['toggle_recording']) and self._can_press_key('toggle_recording'):
                    self.recording = not self.recording
                    if not self.recording and hasattr(self, 'video_writer'):
                        self.video_writer.release()
                        delattr(self, 'video_writer')
                    self._show_notification(
                        "Recording Started" if self.recording else "Recording Stopped",
                        'warning' if self.recording else 'info'
                    )
                    
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
                    
                self.performance_monitor.end_frame()
                
        except KeyboardInterrupt:
            print("\nClosing Programme.")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            print("\nCleaning...")
            # Cleanup video writer if recording
            if hasattr(self, 'video_writer'):
                self.video_writer.release()
            # Save face names
            if self.face_detector and hasattr(self.face_detector, '_save_face_names'):
                self.face_detector._save_face_names()
            # Cleanup performance monitor
            self.performance_monitor.cleanup()
            # Release camera
            cap.release()
            # Close all windows
            cv2.destroyAllWindows()
            print("Programme closed.")

    def _update_translation(self, translation):
        """Update current translation and reset timer"""
        self.current_translation = translation
        self.translation_time = time.time()
        
    def _draw_translation_box(self, frame):
        """Draw translation box with current translation"""
        if not self.current_translation:
            return frame
            
        current_time = time.time()
        if current_time - self.translation_time > self.translation_duration:
            self.current_translation = ""
            return frame
            
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Calculate box dimensions and position
        text_size = cv2.getTextSize(self.current_translation, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        box_width = text_size[0] + 40
        box_height = text_size[1] + 40
        
        x = (width - box_width) // 2
        y = height - box_height - 100
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Draw background box with rounded corners
        cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height),
                     (0, 0, 0), -1)
        
        # Draw text
        text_x = x + 20
        text_y = y + box_height - 20
        cv2.putText(overlay, self.current_translation,
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                   
        # Add fade effect based on time
        alpha = min(1.0, (self.translation_duration - (current_time - self.translation_time)) / 0.5)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
        
    def _handle_keyboard(self):
        """Handle keyboard events with improved key detection"""
        keybinds = self.config['keybinds']
        modes = self.config['modes']
        
        # Mode switching with cooldown
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
            
        # Mouse control
        elif keyboard.is_pressed(keybinds['mouse_control']) and self._can_press_key('mouse_control'):
            if modes['mouse']['enabled']:
                self.active_mode = "mouse" if self.active_mode != "mouse" else "gesture"
                self._show_notification(
                    "Mouse Control Enabled" if self.active_mode == "mouse" else "Mouse Control Disabled",
                    'info'
                )
            
        # Face detection
        elif keyboard.is_pressed(keybinds['face_detection']) and self._can_press_key('face_detection'):
            if modes['face']['enabled']:
                self.active_mode = "face" if self.active_mode != "face" else "gesture"
                self._show_notification(
                    "Face Detection Enabled" if self.active_mode == "face" else "Face Detection Disabled",
                    'info'
                )
            
        # Toggle detection mode (face mesh/iris)
        elif keyboard.is_pressed(keybinds['toggle_detection_mode']) and self._can_press_key('toggle_detection'):
            if self.active_mode == "face" and modes['face']['enabled']:
                self.face_detector.toggle_mode()
                current_mode = self.face_detector.get_mode()
                self._show_notification(f"Face Detection Mode: {current_mode.capitalize()}", 'info')
                
        # Performance display
        elif keyboard.is_pressed(keybinds['toggle_performance']) and self._can_press_key('toggle_performance'):
            self.show_performance = not self.show_performance
            
        # FPS display
        elif keyboard.is_pressed(keybinds['toggle_fps']) and self._can_press_key('toggle_fps'):
            self.show_fps = not self.show_fps
            
        # Help menu
        elif keyboard.is_pressed(keybinds['help_menu']) and self._can_press_key('help_menu'):
            self.show_help = not self.show_help
            
        # Recording
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
        """Show a temporary notification"""
        self.last_notification = (text, type)
        self.notification_time = time.time()
        
    def _update_notification(self, frame):
        """Update and remove notification if needed"""
        if self.last_notification:
            current_time = time.time()
            duration = current_time - self.notification_time
            
            if duration < 2.0:  # Show notification for 2 seconds
                text, type = self.last_notification
                frame = self.ui_manager.add_notification(frame, text, type)
            else:
                self.last_notification = None
                
    def _draw_performance_metrics(self, frame, metrics):
        """Draw performance metrics on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (width-200, 0), (width, 150),
                     (0, 0, 0), -1)
        
        # Add metrics
        metrics_text = [
            f"FPS: {metrics['fps']:.1f}",
            f"Frame Time: {metrics['processing_time']*1000:.1f}ms",
            f"Memory: {metrics['memory_usage']:.1f}%",
            f"CPU: {metrics['cpu_usage']:.1f}%"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(overlay, text, (width-190, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                       
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def _record_frame(self, frame):
        """Record frame to video file"""
        if not hasattr(self, 'video_writer'):
            # Create recordings directory if it doesn't exist
            os.makedirs('recordings', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join('recordings', f'recording_{timestamp}.mp4')
            
            # Get frame properties
            height, width = frame.shape[:2]
            fps = self.fps_counter.get_fps()
            if fps < 10:  
                fps = 30
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
        # Write frame
        self.video_writer.write(frame)

if __name__ == "__main__":
    app = RTTSSL()
    app.run()
