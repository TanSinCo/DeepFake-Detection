# src/test_temporal.py

import torch
import numpy as np

from dataset import DeepfakeDataset
from models.spatial_cnn import SpatialCNN
from models.temporal_lstm import TemporalLSTM


# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = DeepfakeDataset(
    "data/metadata.csv"
)

sample = dataset[0]

frames = sample["frames_15"]

print("Frames shape:", frames.shape)


# -----------------------------
# SELECT FRAMES
# -----------------------------
selected_frames = frames[:60]

print("Selected frames:", selected_frames.shape)


# -----------------------------
# SPATIAL CNN
# -----------------------------
cnn = SpatialCNN()

spatial_features = cnn.extract_features(
    selected_frames
)

print(
    "Spatial feature shape:",
    spatial_features.shape
)


# -----------------------------
# ADD BATCH DIMENSION
# -----------------------------
spatial_features = spatial_features.unsqueeze(0)

print(
    "LSTM input shape:",
    spatial_features.shape
)


# -----------------------------
# TEMPORAL MODEL
# -----------------------------
lstm_model = TemporalLSTM()

temporal_features = lstm_model(
    spatial_features
)

print(
    "Temporal feature shape:",
    temporal_features.shape
)