# src/test_spatial.py

import os
import torch
import numpy as np

from dataset import DeepfakeDataset
from models.spatial_cnn import SpatialCNN


# ---------------------------------
# LOAD DATASET
# ---------------------------------

BASE_DIR = os.path.dirname(__file__)

csv_path = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "metadata.csv"
)

csv_path = os.path.abspath(csv_path)

dataset = DeepfakeDataset(csv_path)

sample = dataset[0]


# ---------------------------------
# LOAD FRAMES
# ---------------------------------

frames = sample["frames_15"]

print("Frames shape:", frames.shape)


# ---------------------------------
# SAMPLE FEW FRAMES
# ---------------------------------

selected_frames = frames[:8]

print("Selected frames:", selected_frames.shape)


# ---------------------------------
# CONVERT TO TENSOR
# ---------------------------------

frames_tensor = torch.tensor(
    selected_frames,
    dtype=torch.float32
)

# Normalize to 0-1
frames_tensor /= 255.0

# HWC → CHW
frames_tensor = frames_tensor.permute(0, 3, 1, 2)

print("Tensor shape:", frames_tensor.shape)


# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = SpatialCNN()

model.eval()


# ---------------------------------
# EXTRACT FEATURES
# ---------------------------------

with torch.no_grad():

    features = model(frames_tensor)

print("Spatial feature shape:", features.shape)