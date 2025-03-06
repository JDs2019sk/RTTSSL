import cv2
import time
import sys
import os
import yaml
from datetime import datetime

def load_config():
    # carrega configurações e keybinds do ficheiro yaml(config/configs.yaml)
    config_path = os.path.join('config', 'configs.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {
            'keybinds': {
                'save_image': 's',
                'reset_camera': 'r',
                'quit_test': 'q' 
            },
            'camera_test': {
                'window_size': {'width': 1280, 'height': 720},
                'save_directory': 'test_images'
            }
        }

def main():
    # carrega as configurações
    config = load_config()
    keybinds = config['keybinds']
    camera_config = config['camera_test']
    
    # criar save directory se ainda nâo existir
    save_dir = camera_config['save_directory']
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nCamera Test Utility")
    print("------------------")
    print(f"Press '{keybinds['save_image']}' to save a test image")
    print(f"Press '{keybinds['reset_camera']}' to reset camera")
    print(f"Press '{keybinds['quit_test']}' to quit")
    
    cap = None
    window_name = "Camera Test"
    
    def init_camera():
        nonlocal cap
        if cap is not None:
            cap.release()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("\nError: Could not open camera")
            return False
            
        # propriedades da camara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['window_size']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['window_size']['height'])
        cap.set(cv2.CAP_PROP_FPS, 30)
        return True
    
    if not init_camera():
        sys.exit(1)
    
    cv2.namedWindow(window_name)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nError: Failed to grab frame")
            break
            
        # mostrar propriedades da camara
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # info overlay
        cv2.putText(frame, f"Resolution: {width}x{height}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(keybinds['quit_test']):
            break
        elif key == ord(keybinds['save_image']):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"test_image_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"\nSaved test image: {filename}")
        elif key == ord(keybinds['reset_camera']):
            print("\nResetting camera...")
            if not init_camera():
                break
    
    # cleanup
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("\nCamera test completed")

if __name__ == "__main__":
    main()
