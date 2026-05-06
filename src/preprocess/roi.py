# src/preprocess/roi.py

import cv2
import numpy as np
# -----------------------------
# LANDMARK INDICES (MediaPipe)
# -----------------------------

ROI_LANDMARKS = {
    "left_cheek_upper": [50, 101, 205, 207],
    "left_cheek_lower": [205, 207, 425, 411],
    "right_cheek_upper": [280, 330, 425, 423],
    "right_cheek_lower": [425, 423, 350, 345],
    "chin": [152, 148, 176, 377, 400],
    "forehead": [10, 338, 297, 332]
}

# -----------------------------
# CREATE POLYGON MASK
# -----------------------------
def create_mask(frame_shape, points):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    points = np.array(points, dtype=np.int32)

    cv2.fillPoly(mask, [points], 255)
    return mask


# -----------------------------
# EXTRACT ROI FROM FRAME
# -----------------------------
def extract_roi(frame, landmarks):
    rois = {}

    for region, indices in ROI_LANDMARKS.items():
        pts = [landmarks[i] for i in indices]

        mask = create_mask(frame.shape, pts)

        roi = cv2.bitwise_and(frame, frame, mask=mask)

        rois[region] = roi

    return rois


# -----------------------------
# PROCESS FULL VIDEO
# -----------------------------
def extract_all_rois(frames, landmarks_all):
    all_rois = []

    for frame, landmarks in zip(frames, landmarks_all):
        rois = extract_roi(frame, landmarks)
        all_rois.append(rois)

    return all_rois