# finger
import cv2
import mediapipe as mp
import pyttsx3
import time

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speaking speed
engine.setProperty('voice', engine.getProperty('voices')[0].id)  # voice type

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Finger tip IDs
tip_ids = [4, 8, 12, 16, 20]

# Speak function
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Finger counting function
def count_fingers(hand_landmarks, img):
    h, w, _ = img.shape
    lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    fingers = []

    if lm_list:
        # Thumb
        if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other four fingers
        for id in range(1, 5):
            if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

    return fingers.count(1)

# Capture from webcam
cap = cv2.VideoCapture(0)

prev_finger_count = -1
screenshot_taken = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    img_height, img_width, _ = img.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            total_fingers = count_fingers(hand_landmarks, img)

            # Display finger count
            cv2.putText(img, f'Fingers: {total_fingers}', (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

            # Speak only if count changes
            if total_fingers != prev_finger_count:
                if 0 <= total_fingers <= 5:
                    speak(f"{total_fingers} finger{'s' if total_fingers != 1 else ''}")
                prev_finger_count = total_fingers
                screenshot_taken = False  # Reset screenshot trigger

            # Take screenshot if 5 fingers open
            if total_fingers == 5 and not screenshot_taken:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                cv2.imwrite(filename, img)
                print(f"Screenshot saved as {filename}")
                screenshot_taken = True

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        # No hand detected message
        cv2.putText(img, "No hand detected", (20, img_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display instructions
    cv2.putText(img, "Show fingers and press 'q' to quit", (20, img_height - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Finger Counter - Project", img)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
hands.close()
cv2.destroyAllWindows()
