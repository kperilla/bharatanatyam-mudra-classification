import os
# For me, without this, it takes a long time to initialize
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
import copy
import itertools
import csv
from tutmodel import KeyPointClassifier

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
        annotated_image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        # Need to mirror handedness since reference frame is mirrored
        # original_label = handedness[0].category_name
        # handedness[0].category_name = "Right" if original_label == "Left" else "Left"
        # cv2.putText(
        #     annotated_image, f"{handedness[0].category_name}",
        #     (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #     FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA
        # )

    return annotated_image

def calc_bounding_rect(image, landmarks):
    # print(type(landmarks))
    # print(dir(landmarks))
    # print(landmarks)
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'yttutorial/tutmodel/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    # if mode == 2 and (0 <= number <= 9):
    #     csv_path = 'model/point_history_classifier/point_history.csv'
    #     with open(csv_path, 'a', newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([number, *point_history_list])
    return

def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    annotated_image = np.copy(image)
    cv2.rectangle(annotated_image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    # print(dir(handedness[0]))
    # print(handedness[0].index)
    # print(handedness[0].score)
    # print(handedness[0].category_name)
    # print(handedness[0].display_name)
    original_label = handedness[0].category_name
    handedness[0].category_name = "Right" if original_label == "Left" else "Left"
    info_text = handedness[0].category_name
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(annotated_image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # if finger_gesture_text != "":
    #     cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    #     cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv2.LINE_AA)

    return annotated_image


base_options = python.BaseOptions(model_asset_path='googletutorial/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

keypoint_classifier = KeyPointClassifier()

with open('yttutorial/tutmodel/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

while cap.isOpened():
    key = cv2.waitKey(10)
    if key == 27:  # ESC
        break
    ret, image = cap.read()
    if not ret:
        break
    debug_image = copy.deepcopy(image)
    mirror_image = cv2.flip(image, 1)  # Mirror display
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mirror_image)

    rgb_frame = cv2.cvtColor(mirror_image, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_frame)
    #  ####################################################################
    # print(dir(detection_result))
    # print(detection_result.hand_landmarks)
    # print(detection_result.handedness)
    annotated_image = mp_image.numpy_view()
    if detection_result.hand_landmarks is not None:
        for hand_landmarks, handedness in zip(detection_result.hand_landmarks,
                                                detection_result.handedness):
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)
            # pre_processed_point_history_list = pre_process_point_history(
            #     debug_image, point_history)
            # Write to the dataset file
            # logging_csv(number, mode, pre_processed_landmark_list,
            #             pre_processed_point_history_list)

            # Hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            annotated_image = draw_info_text(
                annotated_image,
                brect,
                handedness,
                keypoint_classifier_labels[hand_sign_id],
                None,
                # point_history_classifier_labels[most_common_fg_id[0][0]],
            )
            # if hand_sign_id == 2:  # Point gesture
            #     point_history.append(landmark_list[8])
            # else:
            #     point_history.append([0, 0])
    annotated_image = draw_landmarks_on_image(annotated_image, detection_result)
    cv2.imshow('Landmarks', annotated_image)

cap.release()
cv2.destroyAllWindows()
