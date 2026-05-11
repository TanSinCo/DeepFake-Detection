# src/test_fusion.py

import torch
import numpy as np

from dataset import DeepfakeDataset

from models.spatial_cnn import SpatialCNN
from models.temporal_lstm import TemporalLSTM
from models.fusion_model import FusionModel


# -----------------------------
# LOAD DATA
# -----------------------------
dataset = DeepfakeDataset(
    "data/metadata.csv"
)

sample = dataset[0]


# -----------------------------
# SPATIAL + TEMPORAL
# -----------------------------
frames = sample["frames_15"][:60]

cnn = SpatialCNN()

spatial_features = cnn.extract_features(frames)

print(
    "Spatial features:",
    spatial_features.shape
)

spatial_features = spatial_features.unsqueeze(0)

lstm_model = TemporalLSTM()

temporal_features = lstm_model(
    spatial_features
)

print(
    "Temporal features:",
    temporal_features.shape
)


# -----------------------------
# rPPG FEATURES
# -----------------------------
rppg = sample["filtered_rppg"]

rppg_mean = np.mean(rppg, axis=0)

rppg_features = torch.tensor(
    rppg_mean,
    dtype=torch.float32
).unsqueeze(0)

print(
    "rPPG features:",
    rppg_features.shape
)


# -----------------------------
# BLINK FEATURE
# -----------------------------
blink_signal = sample["blink_signal"]

blink_mean = np.mean(blink_signal)

blink_features = torch.tensor(
    [[blink_mean]],
    dtype=torch.float32
)

print(
    "Blink features:",
    blink_features.shape
)


# -----------------------------
# MOTION FEATURE
# -----------------------------
motion_signal = sample["motion_signal"]

motion_mean = np.mean(motion_signal)

motion_features = torch.tensor(
    [[motion_mean]],
    dtype=torch.float32
)

print(
    "Motion features:",
    motion_features.shape
)


# -----------------------------
# FUSION MODEL
# -----------------------------
fusion_model = FusionModel()

output = fusion_model(
    temporal_features,
    rppg_features,
    blink_features,
    motion_features
)

print(
    "Final output shape:",
    output.shape
)

print(
    "Prediction logit:",
    output
)

prob = torch.sigmoid(output)

print("Fake probability:", prob.item())