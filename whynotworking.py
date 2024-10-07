import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands class
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Define screen dimensions
screen_width = 1920
screen_height = 1080

while True:
    # Read frame from video capture
    success, img = cap.read()
    
    # Check if frame was successfully read
    if not success:
        print("Failed to read frame from video capture device.")
        break
    
    # Flip image horizontally
    img = cv2.flip(img, 1)
    
    # Convert image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process image with MediaPipe hands
    results = hands.process(imgRGB)
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
        # Iterate over detected hands
        for handLMs in results.multi_hand_landmarks:
            # Get hand landmarks
            for id, lm in enumerate(handLMs.landmark):
                # Get x and y coordinates of landmark
                x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                
                # Map hand landmark coordinates to screen coordinates
                screen_x = int(x * screen_width / img.shape[1])
                screen_y = int(y * screen_height / img.shape[0])
                
                # Draw dot on image at landmark location
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                
                # Draw dot on screen at mapped landmark location
                cv2.circle(img, (screen_x, screen_y), 5, (0, 0, 255), -1)
    
    # Display image
    cv2.imshow('Image', img)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()