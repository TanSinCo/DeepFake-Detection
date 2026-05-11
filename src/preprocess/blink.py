# src/preprocess/blink.py

import numpy as np


# ---------------------------------
# LEFT + RIGHT EYE LANDMARKS
# MediaPipe FaceMesh indices
# ---------------------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ---------------------------------
# EUCLIDEAN DISTANCE
# ---------------------------------

def euclidean(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


# ---------------------------------
# EAR CALCULATION
# ---------------------------------

def calculate_ear(eye_points):

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)

    horizontal = euclidean(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# ---------------------------------
# EXTRACT EYE POINTS
# ---------------------------------

def get_eye_points(landmarks, indices):

    eye = []

    for idx in indices:
        eye.append(landmarks[idx])

    return eye


# ---------------------------------
# MAIN BLINK FEATURE EXTRACTION
# ---------------------------------

def extract_blink_signal(landmarks_all):

    ear_signal = []

    for landmarks in landmarks_all:

        left_eye = get_eye_points(
            landmarks,
            LEFT_EYE
        )

        right_eye = get_eye_points(
            landmarks,
            RIGHT_EYE
        )

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        avg_ear = (left_ear + right_ear) / 2.0

        ear_signal.append(avg_ear)

    return np.array(ear_signal)