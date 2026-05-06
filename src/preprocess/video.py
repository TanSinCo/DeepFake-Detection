# src/preprocess/video.py

import cv2
import mediapipe as mp
import numpy as np

print("MediaPipe location:", mp.__file__)

FRAME_SIZE = 224

mp_face_mesh = mp.solutions.face_mesh


# -----------------------------
# INIT FACEMESH
# -----------------------------
def init_face_mesh():

    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )


# -----------------------------
# EXTRACT FRAMES + LANDMARKS
# -----------------------------
def extract_frames_and_landmarks(
    video_path,
    target_fps=15,
    max_frames=180
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None, None

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = 30

    frame_skip = max(int(original_fps // target_fps), 1)

    face_mesh = init_face_mesh()

    frames = []
    landmarks_all = []

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # FPS CONTROL
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Resize
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FaceMesh
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            h, w, _ = frame.shape

            lm_array = []

            for lm in results.multi_face_landmarks[0].landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                lm_array.append([x, y])

            frames.append(frame)
            landmarks_all.append(lm_array)

            if len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        return None, None

    return np.array(frames), np.array(landmarks_all)