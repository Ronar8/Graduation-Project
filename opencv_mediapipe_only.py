import math

import cv2
import mediapipe as mp

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def calculate_distance(point_1, point_2):
    if point_1 is None or point_2 is None:
        return 0

    x_1, y_1 = point_1
    x_2, y_2 = point_2
    distance = math.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)

    return distance


def calculate_ratio(eye_top, eye_bottom, eye_left, eye_right):
    vertical_length = calculate_distance(eye_top, eye_bottom)
    horizontal_length = calculate_distance(eye_left, eye_right)

    if vertical_length == 0 or horizontal_length == 0:
        return 100

    eye_ratio = (vertical_length / horizontal_length) * 100

    return eye_ratio


ratio_list_left = []
ratio_list_right = []

# thickness and circle radius of annotations
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# open default camera through opencv
cap = cv2.VideoCapture(0)

frame_counter = 0
blink_counter = 0
eye_shut_time = 0

eye_shut = False
wait_next_5_frames = False

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# capture frames from webcam input
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        # if frame is not read correctly, end stream
        if not success:
            print("Can't receive frame. Ending stream.")
            break

        # change color space from BGR to RGB for frame processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # after frame is processed, draw annotations on the image and
        # return to BGR color space
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

            landmark = face_landmarks.landmark

            left_eye_top = _normalized_to_pixel_coordinates(landmark[386].x, landmark[386].y, width, height)
            left_eye_bottom = _normalized_to_pixel_coordinates(landmark[374].x, landmark[374].y, width, height)
            left_eye_left = _normalized_to_pixel_coordinates(landmark[263].x, landmark[263].y, width, height)
            left_eye_right = _normalized_to_pixel_coordinates(landmark[362].x, landmark[362].y, width, height)

            right_eye_top = _normalized_to_pixel_coordinates(landmark[159].x, landmark[159].y, width, height)
            right_eye_bottom = _normalized_to_pixel_coordinates(landmark[145].x, landmark[145].y, width, height)
            right_eye_left = _normalized_to_pixel_coordinates(landmark[133].x, landmark[133].y, width, height)
            right_eye_right = _normalized_to_pixel_coordinates(landmark[33].x, landmark[33].y, width, height)

            cv2.line(image, left_eye_top, left_eye_bottom, (0, 255, 0), 3)
            cv2.line(image, right_eye_top, right_eye_bottom, (0, 255, 0), 3)

            ratio_left_eye = calculate_ratio(left_eye_top, left_eye_bottom, left_eye_left, left_eye_right)
            ratio_list_left.append(ratio_left_eye)
            if len(ratio_list_left) > 5:
                ratio_list_left.pop(0)
            ratioAvg_left = sum(ratio_list_left) / len(ratio_list_left)

            ratio_right_eye = calculate_ratio(right_eye_top, right_eye_bottom, right_eye_left, right_eye_right)
            ratio_list_right.append(ratio_right_eye)
            if len(ratio_list_right) > 5:
                ratio_list_right.pop(0)
            ratioAvg_right = sum(ratio_list_right) / len(ratio_list_right)

            # add every frame when eye is shut
            if ratioAvg_left < 28 and ratioAvg_right < 28:
                frame_counter += 1

            # every 5 blink, sum up all frames recorded when eye was shut
            if blink_counter % 5 == 0 and blink_counter != 0:
                eye_shut_time = frame_counter
                frame_counter = 0

            if eye_shut_time > 70:
                print("UWAGA! Ryzyko zasniecia wysokie")

            # Adds 1 to blink counter
            if ratioAvg_left < 28 and ratioAvg_right < 28 and eye_shut is False:
                blink_counter += 1
                eye_shut = True

            # Check if users eyes are still shut after detecting blink
            if ratioAvg_left >= 28 and ratioAvg_right >= 28:
                eye_shut = False

            #print(frame_counter)

            cv2.putText(image, f'Blink count: {blink_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f'FPS: {fps}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('facemesh', image)

        # quit on q key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
