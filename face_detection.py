import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Starting webcam feed. Press 'q' to exit.")

# movement detection state
prev_mean_x = None
prev_mean_y = None
current_dir = None  # currently held direction
last_print_time = 0.0
HOLD_INTERVAL = 0.1  # print every 0.3s while holding a direction
last_print_mean_x = None
last_print_mean_y = None
baseline_mean_x = None
baseline_mean_y = None
last_seen_mean_x = None
last_seen_mean_y = None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h,w,_ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            # compute mean position across all landmarks (normalized coords)
            n = len(face_landmarks)
            mean_x = sum([lm.x for lm in face_landmarks]) / n
            mean_y = sum([lm.y for lm in face_landmarks]) / n
            # remember last seen centroid for baseline setting
            last_seen_mean_x = mean_x
            last_seen_mean_y = mean_y
            #1 is nose tip, 0 is forehead center
            nose_tip = face_landmarks[1]
            left_ear = face_landmarks[234]
            right_ear = face_landmarks[454]
            mouth = face_landmarks[13]
            between_eyes = face_landmarks[168]
            x0,y0 = int(nose_tip.x * w), int(nose_tip.y * h)
            x1,y1 = int(left_ear.x * w), int(left_ear.y * h)
            x2,y2 = int(right_ear.x * w), int(right_ear.y * h)
            x3,y3 = int(mouth.x * w), int(mouth.y * h)
            x4,y4 = int(between_eyes.x * w), int(between_eyes.y * h)
            cv2.circle(frame, (x0,y0), 10, (255,0,255), -1)
            cv2.circle(frame, (x1,y1), 10, (0,0,255), -1)
            cv2.circle(frame, (x2,y2), 10, (0,0,255), -1)
            cv2.circle(frame, (x3,y3), 10, (255,0,0), -1)
            cv2.circle(frame, (x4,y4), 10, (255,255,0), -1)
            # Draw white lines between all drawn points
            pts = [(x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    cv2.line(frame, pts[i], pts[j], (255, 255, 255), 2)

            # Draw baseline instruction + status
            status_text = "Baseline: set" if baseline_mean_x is not None else "Baseline: unset - press 's'"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if baseline_mean_x is not None else (0,0,255), 2)

            # movement detection using centroid of all landmarks
            # require user-set baseline to be present before printing directions
            ref_x = baseline_mean_x if baseline_mean_x is not None else None
            ref_y = baseline_mean_y if baseline_mean_y is not None else None

            if ref_x is not None and ref_y is not None:
                dx = mean_x - ref_x
                dy = mean_y - ref_y
                # choose dominant movement (vertical or horizontal)
                abs_dx = abs(dx)
                abs_dy = abs(dy)
                thresh = 0.02
                return_thresh = 0.008  # threshold for returning to baseline to reset current_dir
                direction = None
                
                # reset current_dir when face returns close to baseline
                if max(abs_dx, abs_dy) < return_thresh:
                    current_dir = None
                
                if max(abs_dx, abs_dy) > thresh:
                    if abs_dy >= abs_dx:
                        if dy > thresh:
                            direction = 'forward'   # chin down -> forward
                        elif dy < -thresh:
                            direction = 'backwards' # chin up -> backwards
                    else:
                        if dx < -thresh:
                            direction = 'left'      # moved left
                        elif dx > thresh:
                            direction = 'right'     # moved right

                # print direction: on new direction or periodically while holding same direction
                now = time.time()
                if direction:
                    if direction != current_dir:
                        # new direction detected, print immediately
                        print(direction)
                        current_dir = direction
                        last_print_time = now
                    elif (now - last_print_time > HOLD_INTERVAL):
                        # same direction held, print again after interval
                        print(direction)
                        last_print_time = now

            # always keep prev_mean as most recent seen centroid; when face is lost we'll clear it
            prev_mean_x = mean_x
            prev_mean_y = mean_y
        # end for face_landmarks
    else:
        # no face detected: clear prev centroid so re-appearance won't be compared to stale prev
        prev_mean_x = None
        prev_mean_y = None

    # show baseline hint in terminal on first run
    # handle key presses for baseline setting/reset

    cv2.imshow('face track', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if last_seen_mean_x is not None:
            baseline_mean_x = last_seen_mean_x
            baseline_mean_y = last_seen_mean_y
            print('Baseline set')
        else:
            print('No face detected to set baseline')
    elif key == ord('r'):
        baseline_mean_x = None
        baseline_mean_y = None
        print('Baseline reset')

cap.release()
cv2.destroyAllWindows()