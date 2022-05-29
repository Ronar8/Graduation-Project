import math

import cv2
import mediapipe as mp

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def calculate_distance(point_1, point_2):
    x_1, y_1 = point_1
    x_2, y_2 = point_2
    distance = math.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)

    return distance

ratio_list_left = []

# thickness and circle radius of annotations
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# open default camera through opencv
cap = cv2.VideoCapture(0)

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


        landmarks = results.multi_face_landmarks[0]

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for cor in [263, 362, 386, 374]:
            left_eye = _normalized_to_pixel_coordinates(landmarks.landmark[cor].x, landmarks.landmark[cor].y, width, height)
            # cv2.line(image, landmarks.landmark[386], left_eye_lower_point, (0, 255, 0), 3)
            # cv2.putText(image, 'x', left_eye, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        left_eye_top = _normalized_to_pixel_coordinates(landmarks.landmark[386].x, landmarks.landmark[386].y, width,
                                                     height)
        left_eye_bottom = _normalized_to_pixel_coordinates(landmarks.landmark[374].x, landmarks.landmark[374].y, width,
                                                     height)
        left_eye_left = _normalized_to_pixel_coordinates(landmarks.landmark[263].x, landmarks.landmark[263].y, width,
                                                     height)
        left_eye_right = _normalized_to_pixel_coordinates(landmarks.landmark[362].x, landmarks.landmark[362].y, width,
                                                     height)

        cv2.line(image, left_eye_top, left_eye_bottom, (0, 255, 0), 3)

        vertical_length_left = calculate_distance(left_eye_top, left_eye_bottom)
        horizontal_length_left = calculate_distance(left_eye_left, left_eye_right)
        ratio_left_eye = (vertical_length_left / horizontal_length_left) * 100
        ratio_list_left.append(ratio_left_eye)
        if len(ratio_list_left) > 5:
            ratio_list_left.pop(0)
        ratioAvg_left = sum(ratio_list_left) / len(ratio_list_left)

        print(ratioAvg_left)

        cv2.imshow('facemesh', cv2.flip(image, 1))

        # quit on q key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
