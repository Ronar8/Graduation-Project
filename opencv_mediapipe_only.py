import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

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

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

