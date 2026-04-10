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


METRICS_QUEUE = ['Sikharam', 'Tamarachudam', 'Sarpasirsha', 'Katakamukha_1', 'Tripathaka', 'Mukulam', 'Chandrakala', 'Suchi', 'Simhamukham']


def normalize_hand_scale_position_rotation(hand_landmarks):
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks])

    # Normalize by translation
    pts -= pts[0]

    # Normalize by scale
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale

    # Normalize by rotation: align wrist->middle MCP vector to point straight up (0, -1)
    ref = pts[9]
    angle = np.arctan2(ref[0], -ref[1])

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])

    # Neat operator for dot product: https://www.pythontutorials.net/blog/numpy-rotation-matrix/#use-operator-for-matrix-multiplication
    pts = pts @ R.T

    return pts.flatten()

def normalize_hand_scale_position(hand_landmarks):
    """no rotation, to stay consistent with image."""
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks])
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()

def get_bounding_box(hand_landmarks, img_w, img_h, padding=0.2):
    xs = [lm.x * img_w for lm in hand_landmarks]
    ys = [lm.y * img_h for lm in hand_landmarks]
    x1 = max(0, int(min(xs) - padding * img_w))
    x2 = min(img_w, int(max(xs) + padding * img_w))
    y1 = max(0, int(min(ys) - padding * img_h))
    y2 = min(img_h, int(max(ys) + padding * img_h))
    return x1, y1, x2, y2

def extract_features(results):
    feats = []

    for hand in results.hand_landmarks[:2]:
        feats.append(normalize_hand_scale_position_rotation(hand))

    while len(feats) < 2:
        feats.append(np.zeros(42))

    return np.concatenate(feats)

def extract_frame_features(frame_rgb, landmark_results, image_size=(224, 224)):
    h, w = frame_rgb.shape[:2]

    # Landmarks
    landmark_feats = []
    for hand in landmark_results.hand_landmarks[:2]:
        landmark_feats.append(normalize_hand_scale_position(hand))
    while len(landmark_feats) < 2:
        landmark_feats.append(np.zeros(42))
    landmarks = np.concatenate(landmark_feats).astype(np.float32)

    # Cropped frame
    x1, y1, x2, y2 = get_bounding_box(landmark_results.hand_landmarks[0], w, h)
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    crop = cv2.resize(crop, image_size).astype(np.float32)
    crop = keras.applications.mobilenet_v2.preprocess_input(crop)

    return crop, landmarks

def select_mode(key, mode):
    if key & 0xFF == ord('n'):
        mode = 0
    if key & 0xFF == ord('m'):
        mode = 1
    return mode



class MetricsController:
    def __init__(self):
        self.countdown_start = None
        self.metrics_start = None
        self.countdown_duration = 3
        self.metrics_queue_ix = 0
        self.metrics_results_dict = {}
        self.total_frames_count = 0
        self.valid_landmarks_count = 0
        self.true_positive_classification_count = 0

    def process_countdown(self, frame):
        if self.countdown_start is not None:
            elapsed_in_countdown = time.time() - self.countdown_start
            remaining_in_countdown = self.countdown_duration - elapsed_in_countdown
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
                self.countdown_start = None  # countdown finished, deinitialize

    def start_countdown(self):
        self.countdown_start = time.time()
        self.metrics_queue_ix = 0

    def process_frame(self, mode, frame):
        if mode == 1 and not self.countdown_start and not self.metrics_start:
            self.metrics_start = time.time()
        if self.metrics_start is not None:
            elapsed_in_metrics_countdown = time.time() - self.metrics_start
            if elapsed_in_metrics_countdown < 8:
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
                self.metrics_start = None
                self.countdown_start = time.time()
                valid_landmark_ratio = self.valid_landmarks_count / self.total_frames_count
                true_positive_classification_ratio = self.true_positive_classification_count / self.total_frames_count
                print('total_frames', self.total_frames_count)
                print('TP', self.true_positive_classification_count)
                self.metrics_results_dict[self.metrics_queue_ix] = {
                    'label': METRICS_QUEUE[self.metrics_queue_ix],
                    'valid_landmarks_ratio': valid_landmark_ratio,
                    'true_positive_classification_ratio': true_positive_classification_ratio,
                }
                print(self.metrics_results_dict[self.metrics_queue_ix])
                print()
                total_frames_count = 0
                valid_landmarks_count = 0
                true_positive_classification_count = 0
                self.metrics_queue_ix += 1
                if self.metrics_queue_ix >= len(METRICS_QUEUE):
                    mode = 0
                    self.metrics_queue_ix = 0
                    self.display_final_metrics()
                else:
                    print(self.metrics_queue_ix, METRICS_QUEUE[self.metrics_queue_ix])
        return mode

    def update_true_positive_rate(self, classification):
        if self.metrics_start:
            self.valid_landmarks_count += 1
            if classification == METRICS_QUEUE[self.metrics_queue_ix]:
                self.true_positive_classification_count += 1

    def update_frame_count(self):
            if self.metrics_start:
                self.total_frames_count += 1
    def display_final_metrics(self):
        percent_sum = 0
        for row in self.metrics_results_dict.values():
            print(f'{row['label']}: {row['true_positive_classification_ratio']}')
            percent_sum += row['true_positive_classification_ratio']
        print()
        print(f'Average: {percent_sum/len(self.metrics_results_dict)}')

class LandmarkerController:
    def __init__(self):
        model_path = "googletutorial/hand_landmarker.task"

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect_landmarks(self, image, timestamp):
        results = self.detector.detect_for_video(image, timestamp)
        return results

class ClassifierController:
    def __init__(self, model):
        self.model = model

        with open("labels.json") as f:
            label_to_id = json.load(f)

        self.id_to_label = {v:k for k,v in label_to_id.items()}

    def classify_from_landmarks(self, landmark_results, confidence_threshold=0.6):
        features = extract_features(landmark_results).reshape(1, -1)

        pred = self.model.predict(features, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)
        classification = self.id_to_label[class_id]

        if confidence > confidence_threshold:
            label_text = f"{classification} ({confidence:.2f})"
        else:
            label_text = "Low Confidence"

        return classification, label_text
    
    def classify_from_image_and_landmarks(self, frame_rgb, landmarks, confidence_threshold=0.6):
        crop, landmarks = extract_frame_features(frame_rgb, landmarks)
        if crop is not None:

            # Predict
            img_batch = np.expand_dims(crop, axis=0)                    # (1, 224, 224, 3)
            lm_batch  = np.expand_dims(landmarks, axis=0)      # (1, 84)

            probs      = self.model.predict(
                {"image": img_batch, "landmarks": lm_batch},
                verbose=0
            )[0]
            class_id    = np.argmax(probs)
            confidence = probs[class_id]
            classification = self.id_to_label[class_id]

            if confidence > confidence_threshold:
                label_text = f"{classification} ({confidence:.2f})"
            else:
                label_text = "Low Confidence"

            return classification, label_text



def main():
    # Load model
    # model = keras.models.load_model("mudra_model.keras")
    model = keras.models.load_model("best_model_mobilenet.keras")
    landmarker = LandmarkerController()
    metrics = MetricsController()
    classifier = ClassifierController(model)

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
            metrics.start_countdown()
        # "n" is "normal mode" (0), "m" is "metrics mode" (1)
        mode = select_mode(key, mode)

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = landmarker.detect_landmarks(mp_image, timestamp)
        timestamp += 1

        label_text = "No hand"

        metrics.process_countdown(frame)
        mode = metrics.process_frame(mode, frame)
        metrics.update_frame_count()

        if results.hand_landmarks:
            # classification, label_text = classifier.classify_from_landmarks(results)
            classification, label_text = classifier.classify_from_image_and_landmarks(rgb, results)
            metrics.update_true_positive_rate(classification)

        cv2.putText(frame, label_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Mudra Classifier", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
