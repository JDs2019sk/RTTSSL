import cv2
import yaml
import os

class HelpMenu:
    def __init__(self):
        self.config_path = os.path.join('config', 'configs.yaml')
        self.keybinds = self._load_config()
        
    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                main_keybinds = {
                    'Mode Selection': {
                        config['keybinds']['letter_mode']: 'Letter Mode',
                        config['keybinds']['word_mode']: 'Word Mode',
                        config['keybinds']['gesture_mode']: 'Gesture Mode',
                        config['keybinds']['mouse_control']: 'Mouse Control',
                        config['keybinds']['face_detection']: 'Face Detection',
                        config['keybinds']['toggle_detection_mode']: 'Toggle Face Mode'
                    },
                    'Controls': {
                        config['keybinds']['toggle_fps']: 'Toggle FPS',
                        config['keybinds']['toggle_performance']: 'Toggle Performance',
                        config['keybinds']['toggle_recording']: 'Toggle Recording',
                        config['keybinds']['help_menu']: 'Help Menu',
                        config['keybinds']['exit']: 'Exit'
                    }
                }
                return main_keybinds
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {
                'Mode Selection': {
                    '1': 'Letter Mode',
                    '2': 'Word Mode',
                    '3': 'Gesture Mode',
                    'm': 'Mouse Control',
                    'f': 'Face Detection',
                    'e': 'Toggle Face Mode'
                },
                'Controls': {
                    'tab': 'Toggle FPS',
                    'p': 'Toggle Performance',
                    'r': 'Toggle Recording',
                    'h': 'Help Menu',
                    'esc': 'Exit'
                }
            }

    def draw(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (600, 460), (0, 0, 0), -1)
        
        cv2.putText(overlay, "Help Menu", (70, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        y_pos = 130

        cv2.putText(overlay, "Mode Selection:", (70, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        
        for key, action in self.keybinds['Mode Selection'].items():
            text = f"{key.upper()}: {action}"
            cv2.putText(overlay, text, (90, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25
            
        y_pos += 10
        
        cv2.putText(overlay, "Controls:", (70, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        
        for key, action in self.keybinds['Controls'].items():
            text = f"{key.upper()}: {action}"
            cv2.putText(overlay, text, (90, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25
        
        alpha = 0.7
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
