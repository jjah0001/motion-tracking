import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

x,y= 0,0

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2,
                                        min_hand_detection_confidence=0.8,
                                        min_hand_presence_confidence=0.8,
                                        min_tracking_confidence=0.8)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Starting webcam feed. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h,w,_ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            base_tip = hand_landmarks[0]
            index_tip = hand_landmarks[8]
            x0,y0 = int(base_tip.x * w), int(base_tip.y * h)
            x1,y1 = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (x0,y0), 10, (255,0,255), -1)
            cv2.circle(frame, (x1,y1), 10, (0,255,255), -1)

    cv2.imshow('index finger track', frame)
    #print("Index finger tip coordinates: ", (x,y))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release
cv2.destroyAllWindows()