# src/preprocess/window.py

import numpy as np


# -----------------------------------
# CREATE SLIDING WINDOWS
# -----------------------------------

def create_windows(signal, window_size=180, stride=90):

    """
    signal shape:
    (T, C)

    Example:
    (360, 5)

    Output:
    (num_windows, window_size, C)
    """

    windows = []

    total_frames = len(signal)

    for start in range(
        0,
        total_frames - window_size + 1,
        stride
    ):

        end = start + window_size

        window = signal[start:end]

        windows.append(window)

    return np.array(windows)