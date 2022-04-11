import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# Capture video from webcam (0 - default), resolution: 1920x1080
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize detector and plot for blinking
detector = FaceMeshDetector(maxFaces=1)
blink_plot = LivePlot(640, 360, [0, 52], invert=True)

# Initialize FPS counter
fpsReader = cvzone.FPS()

# List to highlight eye landmarks with purple color
left_eye_landmarks = [33, 7, 246, 163, 161, 144, 160, 145, 159, 153, 158, 154, 157, 155, 173, 133]
right_eye_landmarks = [263, 249, 466, 390, 388, 373, 387, 374, 386, 380, 385, 381, 384, 382, 398, 362]
landmarks_color = (255, 0, 255)

# List for smoothing out ratio (distance between vertical and horizontal landmarks) on the plot
ratio_list_left = []
ratio_list_right = []

frame_list = []

blink_counter = 0
frame_counter = 0
eye_shut_time = 0

eye_shut = False
wait_next_5_frames = False

while True:
    success, img = cap.read()
    fps, img = fpsReader.update(img, pos=(50, 30), color=(0, 255, 0), scale=2, thickness=2)
    img, faces = detector.findFaceMesh(img)

    if faces:
        face = faces[0]
        for point in left_eye_landmarks:
            cv2.circle(img, face[point], 5, landmarks_color, cv2.FILLED)
        for point in right_eye_landmarks:
            cv2.circle(img, face[point], 5, landmarks_color, cv2.FILLED)

        # Define relevant landmarks
        left_eye_upper_point = face[159]
        left_eye_lower_point = face[145]
        left_eye_leftmost_point = face[33]
        left_eye_rightmost_point = face[133]
        right_eye_upper_point = face[386]
        right_eye_lower_point = face[374]
        right_eye_leftmost_point = face[362]
        right_eye_rightmost_point = face[263]

        # Place lines between defined points
        cv2.line(img, left_eye_upper_point, left_eye_lower_point, (0, 255, 0), 3)
        cv2.line(img, left_eye_leftmost_point, left_eye_rightmost_point, (0, 255, 0), 3)

        # Calculate ratio between vertical and horizontal length for left eye
        vertical_length_left, _ = detector.findDistance(left_eye_upper_point, left_eye_lower_point)
        horizontal_length_left, _ = detector.findDistance(left_eye_leftmost_point, left_eye_rightmost_point)
        ratio_left_eye = (vertical_length_left / horizontal_length_left) * 100
        ratio_list_left.append(ratio_left_eye)
        if len(ratio_list_left) > 5:
            ratio_list_left.pop(0)
        ratioAvg_left = sum(ratio_list_left) / len(ratio_list_left)

        # Calculate ratio between vertical and horizontal length for right eye
        vertical_length_right, _ = detector.findDistance(left_eye_upper_point, left_eye_lower_point)
        horizontal_length_right, _ = detector.findDistance(left_eye_leftmost_point, left_eye_rightmost_point)
        ratio_right_eye = (vertical_length_right / horizontal_length_right) * 100
        ratio_list_right.append(ratio_right_eye)
        if len(ratio_list_right) > 5:
            ratio_list_right.pop(0)
        ratioAvg_right = sum(ratio_list_right) / len(ratio_list_right)

        if ratioAvg_left < 28 and ratioAvg_right < 28:
            frame_counter += 1

        if blink_counter % 5 != 0:
            wait_next_5_frames = False

        if blink_counter % 5 == 0 and blink_counter != 0 and wait_next_5_frames is False:
            eye_shut_time = frame_counter
            frame_counter = 0
            wait_next_5_frames = True

        if eye_shut_time > 50:
            print("UWAGA! Ryzyko zasniecia wysokie")

        # Adds 1 to counter and changes color when user blinks
        if ratioAvg_left < 28 and ratioAvg_right < 28 and eye_shut is False:
            blink_counter += 1
            landmarks_color = (0, 255, 0)
            eye_shut = True

        # Check if users eyes are still shut after first frame of blink
        if ratioAvg_left >= 28:
            landmarks_color = (255, 0, 255)
            eye_shut = False

        # Display blink counter, top-left position
        cvzone.putTextRect(img, f'Blink count: {blink_counter}', (50, 80))
        cvzone.putTextRect(img, f'Eye-shut time: {eye_shut_time}', (50, 160))

        ratioAvg = (ratioAvg_left + ratioAvg_right) / 2

        # Display plot next to registered video
        img_plot = blink_plot.update(ratioAvg)
        img = cv2.resize(img, (640, 360))
        img_stack = cvzone.stackImages([img, img_plot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        img_stack = cvzone.stackImages([img, img], 2, 1)

    # Show result in Video window, close on 'Q' key press
    cv2.imshow("Video", img_stack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
