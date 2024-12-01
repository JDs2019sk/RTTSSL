"""
Help Menu Module
Provides functionality to display an interactive help menu showing all available
keybinds and their functions.
"""

import cv2
import yaml
import os

class HelpMenu:
    def __init__(self):
        self.config_path = os.path.join('config', 'keybinds.yaml')
        self.keybinds = self._load_config()
        
    def _load_config(self):
        """Load keybinds configuration from yaml file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading keybinds config: {e}")
            return {}
            
    def draw(self, frame):
        """Draw help menu overlay on the frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (600, 500), (0, 0, 0), -1)
        
        # Add title
        cv2.putText(overlay, "Help Menu", (70, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add keybinds
        y_pos = 130
        for action, key in self.keybinds.get('keybinds', {}).items():
            text = f"{key.upper()}: {action.replace('_', ' ').title()}"
            cv2.putText(overlay, text, (70, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 30
            
        # Blend overlay with original frame
        alpha = 0.7
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
