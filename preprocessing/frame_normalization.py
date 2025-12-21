"""
Normalizes skeleton sequences to a fixed length (10 frames)
using motion-based selection and temporal interpolation.
"""

import os
import json
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path


def load_keypoints(json_path):
    with open(json_path) as f:
        data = json.load(f)

    people = data.get("people", [])
    frames = np.zeros((2, 25, 3))

    for i in range(min(2, len(people))):
        kp = np.array(people[i]["pose_keypoints_2d"]).reshape(25, 3)
        frames[i] = kp

    return frames


def interpolate_missing_joints(sequence):
    seq = sequence.copy()
    for t in range(1, len(seq) - 1):
        for p in range(2):
            for j in range(25):
                if seq[t, p, j, 2] == 0:
                    seq[t, p, j] = (seq[t - 1, p, j] + seq[t + 1, p, j]) / 2
    return seq


def select_active_frames(sequence, target_frames=10):
    if len(sequence) <= target_frames:
        return temporal_interpolation(sequence, target_frames)

    motion = []
    for i in range(len(sequence) - 1):
        diff = np.sum(np.abs(sequence[i + 1, :, :, :2] - sequence[i, :, :, :2]))
        motion.append(diff)

    idx = np.argsort(motion)[-target_frames:]
    return sequence[idx]


def temporal_interpolation(sequence, target_frames=10):
    x_old = np.linspace(0, 1, len(sequence))
    x_new = np.linspace(0, 1, target_frames)

    result = np.zeros((target_frames, 2, 25, 3))
    for p in range(2):
        for j in range(25):
            for c in range(3):
                f = interp1d(x_old, sequence[:, p, j, c], kind="linear")
                result[:, p, j, c] = f(x_new)

    return result


def process_video_json(input_dir, output_dir, target_frames=10):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    sequence = [load_keypoints(f) for f in json_files]
    sequence = np.array(sequence)

    sequence = interpolate_missing_joints(sequence)
    sequence = select_active_frames(sequence, target_frames)

    video_out = output_dir / input_dir.name
    video_out.mkdir(exist_ok=True)

    for i, frame in enumerate(sequence):
        out_json = {
            "version": 1.3,
            "people": []
        }
        for p in range(2):
            if np.any(frame[p]):
                out_json["people"].append({
                    "pose_keypoints_2d": frame[p].flatten().tolist()
                })

        with open(video_out / f"frame_{i:04d}_keypoints.json", "w") as f:
            json.dump(out_json, f)
