import cv2
import time
import sys

def test_camera():
    print("\n=== Camera Test ===")
    print("Initializing camera...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("[X] Error: Could not open camera")
        print("Please check:")
        print("1. Camera is properly connected")
        print("2. No other application is using the camera")
        print("3. Camera permissions are enabled")
        return False
    
    print("[+] Camera opened successfully!")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save a test image")
    print("- Press 'r' to reset camera")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[X] Error: Could not read frame")
            print("Attempting to reset camera...")
            cap.release()
            cap = cv2.VideoCapture(0)
            continue
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            # Draw FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_count = 0
            start_time = time.time()
        
        # Show frame dimensions
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Resolution: {width}x{height}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[+] Camera test completed successfully")
            break
        elif key == ord('s'):
            # Save test image
            filename = f"camera_test_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[+] Saved test image: {filename}")
        elif key == ord('r'):
            print("[*] Resetting camera...")
            cap.release()
            cap = cv2.VideoCapture(0)
    
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    try:
        test_camera()
    except Exception as e:
        print(f"\n[X] Error: {str(e)}")
        print("Camera test failed. Please check your camera connection and try again.")
        sys.exit(1)
