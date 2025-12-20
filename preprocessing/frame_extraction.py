"""
Video Frame Extraction Utility
Used as a preprocessing step before skeleton extraction.
"""

import cv2
import os


def extract_frames(video_path, output_dir, fps=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    count = 0
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % fps == 0:
            cv2.imwrite(
                os.path.join(output_dir, f"frame_{count:04d}.jpg"),
                frame
            )
            count += 1

        frame_id += 1

    cap.release()
