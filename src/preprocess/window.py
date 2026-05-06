# src/preprocess/window.py

import numpy as np


# -----------------------------------
# CREATE TEMPORAL WINDOWS
# -----------------------------------
def create_windows(
    signal,
    window_size=180,
    step_size=90
):

    windows = []

    total_frames = len(signal)

    for start in range(
        0,
        total_frames - window_size + 1,
        step_size
    ):

        end = start + window_size

        window = signal[start:end]

        windows.append(window)

    windows = np.array(windows)

    return windows