# src/dataset.py

import pandas as pd

from preprocess.video import extract_frames_and_landmarks
from preprocess.roi import extract_all_rois
from preprocess.rppg import extract_rppg
from preprocess.filtering import filter_rppg
from preprocess.window import create_windows


class DeepfakeDataset:

    def __init__(self, csv_path):

        self.data = pd.read_csv(csv_path)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        video_path = row["video_path"]
        label = row["label"]

        # -----------------------------------
        # 15 FPS BRANCH
        # -----------------------------------
        frames_15, landmarks_15 = extract_frames_and_landmarks(
            video_path,
            target_fps=15,
            max_frames=180
        )

        # -----------------------------------
        # 30 FPS rPPG BRANCH
        # -----------------------------------
        frames_30, landmarks_30 = extract_frames_and_landmarks(
            video_path,
            target_fps=30,
            max_frames=360
        )

        if frames_15 is None or frames_30 is None:
            print(f"⚠️ Skipping bad video: {video_path}")
            return None

        # -----------------------------------
        # ROI EXTRACTION
        # -----------------------------------
        rois = extract_all_rois(frames_30, landmarks_30)

        # -----------------------------------
        # rPPG EXTRACTION
        # -----------------------------------
        rppg = extract_rppg(rois)

        # -----------------------------------
        # FILTERING
        # -----------------------------------
        filtered_rppg = filter_rppg(rppg)
        rppg_windows = create_windows(
            filtered_rppg,
            window_size=180,
            step_size=90
        )

        return {

            "frames_15": frames_15,
            "landmarks_15": landmarks_15,

            "frames_30": frames_30,
            "landmarks_30": landmarks_30,

            "rois": rois,

            "rppg": rppg,
            "filtered_rppg": filtered_rppg,
            "rppg_windows": rppg_windows,

            "label": label
        }