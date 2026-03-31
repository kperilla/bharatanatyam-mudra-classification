import os
# For me, without this, it takes a long time to initialize
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow import keras
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python

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

# Camera
cap = cv2.VideoCapture(0)
timestamp = 0

def extract_features(results):
    feats = []

    for hand in results.hand_landmarks[:2]:
        for lm in hand:
            feats.extend([lm.x, lm.y, lm.z])

    while len(feats) < 126:
        feats.extend([0]*63)

    return np.array(feats)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()