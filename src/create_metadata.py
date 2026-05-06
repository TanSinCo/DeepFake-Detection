import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ffpp")
BASE_DIR = os.path.abspath(BASE_DIR)

def create_metadata():
    video_paths = []
    labels = []

    for label_name in ["real", "fake"]:
        folder_path = os.path.join(BASE_DIR, label_name)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        label = 0 if label_name == "real" else 1

        for file in os.listdir(folder_path):
            if file.endswith(".mp4"):
                full_path = os.path.join(folder_path, file)

                video_paths.append(full_path)
                labels.append(label)

    df = pd.DataFrame({
        "video_path": video_paths,
        "label": labels
    })

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.csv")
    output_path = os.path.abspath(output_path)

    df.to_csv(output_path, index=False)

    print("✅ Metadata created successfully!")
    print(f"Saved at: {output_path}")

if __name__ == "__main__":
    create_metadata()