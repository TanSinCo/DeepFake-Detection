# src/test_data.py

import os
import matplotlib.pyplot as plt

from dataset import DeepfakeDataset


# -----------------------------------
# CSV PATH
# -----------------------------------
BASE_DIR = os.path.dirname(__file__)

csv_path = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "metadata.csv"
)

csv_path = os.path.abspath(csv_path)


# -----------------------------------
# DATASET
# -----------------------------------
dataset = DeepfakeDataset(csv_path)

print("Total samples:", len(dataset))


# -----------------------------------
# SAMPLE
# -----------------------------------
sample = dataset[0]

if sample is None:

    print("❌ Sample failed to load")

    exit()


# -----------------------------------
# SHAPES
# -----------------------------------
print()

print("15 FPS Frames:",
      sample["frames_15"].shape)

print("30 FPS Frames:",
      sample["frames_30"].shape)

print()

print("15 FPS Landmarks:",
      sample["landmarks_15"].shape)

print("30 FPS Landmarks:",
      sample["landmarks_30"].shape)

print()

print("ROI Regions:",
      sample["rois"][0].keys())

print()

print("rPPG Shape:",
      sample["rppg"].shape)

print("Filtered rPPG Shape:",
      sample["filtered_rppg"].shape)

print("rPPG Windows Shape:",
      sample["rppg_windows"].shape)

print("Blink shape:", sample["blink_signal"].shape)

print("Motion shape:", sample["motion_signal"].shape)

print()

print("Label:",
      sample["label"])


# -----------------------------------
# VISUALIZATION
# -----------------------------------
plt.figure(figsize=(12, 5))

raw = sample["rppg"][:, 0]

raw = (raw - raw.mean()) / raw.std()

filtered = sample["filtered_rppg"][:, 0]

plt.plot(raw, label="Raw POS")

plt.plot(filtered, label="Filtered POS")

plt.legend()

plt.title("Raw vs Filtered rPPG (POS Method)")

plt.xlabel("Frame")

plt.ylabel("Signal")

plt.figure(figsize=(12, 4))

plt.plot(sample["blink_signal"])

plt.title("Blink EAR Signal")

plt.xlabel("Frame")

plt.ylabel("EAR")

plt.figure(figsize=(12, 4))

plt.plot(sample["motion_signal"])

plt.title("Micro Motion Signal")

plt.xlabel("Frame")

plt.ylabel("Motion Magnitude")

plt.show()