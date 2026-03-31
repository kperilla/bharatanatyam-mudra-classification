import os
# For me, without this, it takes a long time to initialize
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow import keras
import time
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python


METRICS_QUEUE = ['Sikharam', 'Tamarachudam', 'Sarpasirsha', 'Katakamukha_1', 'Tripathaka', 'Mukulam', 'Chandrakala']


def extract_features(results):
    feats = []

    for hand in results.hand_landmarks[:2]:
        for lm in hand:
            feats.extend([lm.x, lm.y, lm.z])

    while len(feats) < 126:
        feats.extend([0]*63)

    return np.array(feats)

def select_mode(key, mode):
    if key & 0xFF == ord('n'):
        mode = 0
    if key & 0xFF == ord('m'):
        mode = 1
    return mode


def process_countdown(countdown_start, countdown_duration, frame):
    if countdown_start is not None:
        elapsed_in_countdown = time.time() - countdown_start
        remaining_in_countdown = countdown_duration - elapsed_in_countdown
        if remaining_in_countdown > 0:
            number = int(remaining_in_countdown) + 1  # Displays 3, 2, 1
            cv2.putText(
                frame,
                str(number),
                (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (0, 255, 0),
                8,
            )
        else:
            countdown_start = None  # countdown finished, deinitialize
    return countdown_start

def main():
    # Load model
    model = keras.models.load_model("mudra_model.keras")

    with open("labels.json") as f:
        label_to_id = json.load(f)

    id_to_label = {v:k for k,v in label_to_id.items()}

    # MediaPipe
    model_path = "googletutorial/hand_landmarker.task"

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO
    )

    detector = vision.HandLandmarker.create_from_options(options)

    countdown_start = None
    metrics_start = None
    countdown_duration = 3
    metrics_queue_ix = 0

    # Camera
    cap = cv2.VideoCapture(0)
    timestamp = 0
    mode = 0
    while True:
        # Process Key (ESC: end) #################################################
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):  # ESC
            break
        # Start timer for metrics
        if key & 0xFF == ord('m'):
            countdown_start = time.time()

        # "n" is "normal mode" (0), "m" is "metrics mode" (1)
        mode = select_mode(key, mode)

        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        countdown_start = process_countdown(countdown_start, countdown_duration, frame)

        # Start tracking metrics
        if mode == 1 and not countdown_start and not metrics_start:
            metrics_start = time.time()
        if metrics_start is not None:
            elapsed_in_metrics_countdown = time.time() - metrics_start
            if elapsed_in_metrics_countdown < 10:
                number = int(elapsed_in_metrics_countdown)
                cv2.putText(
                    frame,
                    str(number),
                    (frame.shape[1] * 3 // 4 - 30, frame.shape[0] * 3 // 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (0, 255, 0),
                    8,
                )
            else:
                elapsed_in_metrics_countdown = 0
                metrics_start = None
                countdown_start = time.time()
                metrics_queue_ix += 1
                print(metrics_queue_ix)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = detector.detect_for_video(mp_image, timestamp)
        timestamp += 1

        label_text = "No hand"

        if results.hand_landmarks:
            features = extract_features(results).reshape(1, -1)

            pred = model.predict(features, verbose=0)
            class_id = np.argmax(pred)
            confidence = np.max(pred)

            if confidence > 0.6:
                label_text = f"{id_to_label[class_id]} ({confidence:.2f})"

        cv2.putText(frame, label_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Mudra Classifier", frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()