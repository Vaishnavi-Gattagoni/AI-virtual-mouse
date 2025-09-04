import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import subprocess
import time  # To track pinch timing


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen resolution
screen_w, screen_h = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Track last pinch time
last_click_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Mirror effect for natural control
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Get important fingertip coordinates
            index_finger = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP]

            # Convert coordinates to screen size
            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
            middle_x, middle_y = int(middle_finger.x * w), int(middle_finger.y * h)
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)

            # Move cursor to index finger position
            screen_x = np.interp(index_x, [0, w], [0, screen_w])
            screen_y = np.interp(index_y, [0, h], [0, screen_h])
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            # Calculate distances between fingers
            distance_index_thumb = np.linalg.norm([index_x - thumb_x, index_y - thumb_y])
            distance_middle_thumb = np.linalg.norm([middle_x - thumb_x, middle_y - thumb_y])

            current_time = time.time()

            # Left-click on index & thumb pinch
            if distance_index_thumb < 30:
                if current_time - last_click_time < 0.5:  # Detect double pinch within 0.4s
                    pyautogui.doubleClick()
                else:
                    pyautogui.click()
                last_click_time = current_time  # Update last click time

            # Right-click on middle & thumb pinch
            if distance_middle_thumb < 30:
                pyautogui.rightClick()
                pyautogui.sleep(0.2)  # Prevent multiple rapid clicks

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Paint Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
