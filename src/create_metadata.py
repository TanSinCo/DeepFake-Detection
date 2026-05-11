import os
import pandas as pd


# =====================================================
# DATASET ROOTS
# =====================================================

DATASETS = [
    {
        "real_dir": "data/raw/ffpp/real",
        "fake_dir": "data/raw/ffpp/fake"
    },

    # ADD MORE DATASETS HERE LATER

    # {
    #     "real_dir": "data/raw/celebdf/real",
    #     "fake_dir": "data/raw/celebdf/fake"
    # },
]


# =====================================================
# SUPPORTED VIDEO FORMATS
# =====================================================

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)


# =====================================================
# CREATE METADATA
# =====================================================

metadata = []

for dataset in DATASETS:

    real_dir = dataset["real_dir"]
    fake_dir = dataset["fake_dir"]

    # =================================================
    # REAL VIDEOS
    # =================================================

    if os.path.exists(real_dir):

        for file_name in os.listdir(real_dir):

            if file_name.lower().endswith(VIDEO_EXTENSIONS):

                full_path = os.path.join(
                    real_dir,
                    file_name
                )

                metadata.append({
                    "video_path": full_path.replace("\\", "/"),
                    "label": 0
                })

    else:
        print(f"[WARNING] Missing folder: {real_dir}")

    # =================================================
    # FAKE VIDEOS
    # =================================================

    if os.path.exists(fake_dir):

        for file_name in os.listdir(fake_dir):

            if file_name.lower().endswith(VIDEO_EXTENSIONS):

                full_path = os.path.join(
                    fake_dir,
                    file_name
                )

                metadata.append({
                    "video_path": full_path.replace("\\", "/"),
                    "label": 1
                })

    else:
        print(f"[WARNING] Missing folder: {fake_dir}")


# =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame(metadata)

os.makedirs("data", exist_ok=True)

csv_path = "data/metadata.csv"

df.to_csv(csv_path, index=False)


# =====================================================
# SUMMARY
# =====================================================

real_count = (df["label"] == 0).sum()
fake_count = (df["label"] == 1).sum()

print("\n===================================")

print("Metadata CSV created successfully.")

print(f"Saved to: {csv_path}")

print(f"Total videos: {len(df)}")

print(f"Real videos: {real_count}")

print(f"Fake videos: {fake_count}")

print("===================================\n")  