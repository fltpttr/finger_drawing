import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
bg = np.full(shape=(800, 800, 3), fill_value=255, dtype='uint8')
x, y = 0, 0
prev_x, prev_y = 0, 0
z = 50

def nothing(p):
    pass

cv2.namedWindow('bg')
cv2.createTrackbar('red', 'bg', 0, 255, nothing)
cv2.createTrackbar('green', 'bg', 0, 255, nothing)
cv2.createTrackbar('blue', 'bg', 0, 255, nothing)
cv2.createTrackbar('size', 'bg', 0, 255, nothing)
cv2.createTrackbar('blur', 'bg', 0, 255, nothing)
r, g, b = 0, 0, 0
size, blur = 10, 20

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            h, w, _ = bg.shape # image.shape
            for hand_landmarks in results.multi_hand_landmarks:
                x = w - 2 * int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                y = 2 * int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                if x < 0:
                    x = 0
                elif x > 630:
                    x = 630
                if y < 0:
                    y = 0
                elif y > 630:
                    y = 630
                cv2.circle(image, (x, y), 12, (0, 0, 255), -1)

        pts = np.array([[prev_x, prev_y], [x, y]], dtype='int32')
        pts = pts.reshape((-1, 1, 2))
        r = cv2.getTrackbarPos('red', 'bg')
        g = cv2.getTrackbarPos('green', 'bg')
        b = cv2.getTrackbarPos('blue', 'bg')
        size = cv2.getTrackbarPos('size', 'bg')
        blur = cv2.getTrackbarPos('blur', 'bg')
        cv2.polylines(bg, [pts], True, (b, g, r), size)

        if blur < 3:
            blur = 3

        bg = cv2.blur(bg, (blur, blur))

        # if blur % 2 != 0:
        #     bg = cv2.medianBlur(bg, blur)
        # else:
        #     blur += 1

        prev_x = x
        prev_y = y

        image = cv2.flip(image, 1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('z'):
            bg = np.full(shape=(800, 800, 3), fill_value=255, dtype='uint8')

        cv2.imshow('frame', image)
        cv2.imshow('bg', bg)

cap.release()
cv2.destroyAllWindows()
