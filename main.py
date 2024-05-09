'''
The goal is to demonstrate gaze tracking using a live webcam, 
where the movement of the dot on the screen is controlled by the user's gaze direction.

- Initialize necessary libraries and setup the Pygame window to match the screen size.
- `calibrate()` function helps establish a baseline gaze vector representing the center of the screen. 
This calibration involves instructing the user to look at the center and press `SPACE` to capture the initial gaze vector.
- Calculate the gaze vector based on the positions of the left and right eye landmarks. 
- Subtract the calibrated gaze vector (center gaze vector) from the smoothed gaze vector to determine the relative movement direction. 
The dot on the screen is then updated accordingly.
- Allow recalibration of the gaze vector by pressing `SPACE` during the main loop.
- Required `shape_predictor_68_face_landmarks.dat` file
'''


import pygame
import numpy as np
import dlib
import cv2
import pyautogui

pygame.init()
screen_width, screen_height = pyautogui.size()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Gaze-Controlled Dot")
white = (255, 255, 255)
red = (255, 0, 0)
dot_radius = 10

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def calibrate():
    print("Calibration Phase: Look at the center of the screen and press SPACE.")
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            left_eye_center = np.mean(left_eye_pts, axis=0)
            right_eye_center = np.mean(right_eye_pts, axis=0)
            center_gaze_vector = right_eye_center - left_eye_center
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    print("Calibration Complete")
                    return center_gaze_vector
                if event.type == pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    exit()  

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

center_gaze_vector = calibrate()

smoothing_factor = 0.5   
smoothed_gaze_vector = None

# Main Loop
dot_position = [screen_width // 2, screen_height // 2]
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                center_gaze_vector = calibrate()

    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        left_eye_center = np.mean(left_eye_pts, axis=0)
        right_eye_center = np.mean(right_eye_pts, axis=0)
        gaze_vector = right_eye_center - left_eye_center

        if smoothed_gaze_vector is not None:
            smoothed_gaze_vector = smoothing_factor * gaze_vector + (1 - smoothing_factor) * smoothed_gaze_vector
        else:
            smoothed_gaze_vector = gaze_vector

        calibrated_gaze_vector = smoothed_gaze_vector - center_gaze_vector  

        movement_scale = 20 
        dot_movement = calibrated_gaze_vector * movement_scale

        dot_position[0] += int(dot_movement[0]) 
        dot_position[1] -= int(dot_movement[1]) 

        dot_position[0] = max(dot_radius, min(dot_position[0], screen_width - dot_radius))
        dot_position[1] = max(dot_radius, min(dot_position[1], screen_height - dot_radius))

    screen.fill(white)
    pygame.draw.circle(screen, red, dot_position, dot_radius)
    pygame.display.update()

cap.release()
pygame.quit()
