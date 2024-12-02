import cv2

def test_camera():
    print("Testing Camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error")
        return
    
    print("Camera ok!")
    print("(Q) Quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: frames ")
            break
            
        cv2.imshow('Camera test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test end")

if __name__ == "__main__":
    test_camera()
