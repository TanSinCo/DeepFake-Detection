# src/preprocess/rppg.py

import numpy as np


# -----------------------------
# EXTRACT MEAN RGB
# -----------------------------
def extract_mean_rgb(roi):

    mask = np.any(roi > 0, axis=2)

    pixels = roi[mask]

    if len(pixels) == 0:
        return np.array([0, 0, 0])

    return np.mean(pixels, axis=0)


# -----------------------------
# POS METHOD
# -----------------------------
def pos_method(rgb_signal):

    rgb_signal = np.asarray(rgb_signal)

    mean_rgb = np.mean(rgb_signal, axis=0)

    normalized = rgb_signal / (mean_rgb + 1e-6) - 1

    X = normalized[:, 0] - normalized[:, 1]

    Y = (
        normalized[:, 0]
        + normalized[:, 1]
        - 2 * normalized[:, 2]
    )

    alpha = np.std(X) / (np.std(Y) + 1e-6)

    pulse = X - alpha * Y

    return pulse


# -----------------------------
# MAIN rPPG EXTRACTION
# -----------------------------
def extract_rppg(rois_all):

    region_names = [

        "left_cheek_upper",
        "left_cheek_lower",

        "right_cheek_upper",
        "right_cheek_lower",

        "forehead"
    ]

    rppg_signals = []

    for region in region_names:

        rgb_series = []

        for frame_rois in rois_all:

            roi = frame_rois[region]

            mean_rgb = extract_mean_rgb(roi)

            rgb_series.append(mean_rgb)

        rgb_series = np.array(rgb_series)

        # POS METHOD
        signal = pos_method(rgb_series)

        rppg_signals.append(signal)

    # (T, 5)
    rppg_signals = np.stack(rppg_signals, axis=1)

    return rppg_signals