# src/preprocess/motion.py

import numpy as np


# -----------------------------------
# MOTION SIGNAL EXTRACTION
# -----------------------------------

def extract_motion_signal(landmarks):

    """
    landmarks shape:
    (T, 478, 2)

    Output:
    (T-1,)
    """

    motion_signal = []

    for i in range(1, len(landmarks)):

        prev_landmarks = landmarks[i - 1]

        curr_landmarks = landmarks[i]

        displacement = np.linalg.norm(
            curr_landmarks - prev_landmarks,
            axis=1
        )

        mean_motion = np.mean(displacement)

        motion_signal.append(mean_motion)

    return np.array(motion_signal)