"""
Normalizes skeleton sequences to a fixed number of frames (default = 10).
Uses motion-based selection and temporal interpolation.
"""

import json
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d


def load_frame(json_path):
    with open(json_path) as f:
        data = json.load(f)

    frame = np.zeros((2, 25, 3))
    people = data.get("people", [])

    for i in range(min(2, len(people))):
        frame[i] = np.array(
            people[i]["pose_keypoints_2d"]
        ).reshape(25, 3)

    return frame


def interpolate_missing(sequence):
    seq = sequence.copy()
    for t in range(1, len(seq) - 1):
        mask = seq[t, :, :, 2] == 0
        seq[t][mask] = (seq[t - 1][mask] + seq[t + 1][mask]) / 2
    return seq


def temporal_interpolation(sequence, target_frames):
    x_old = np.linspace(0, 1, len(sequence))
    x_new = np.linspace(0, 1, target_frames)

    out = np.zeros((target_frames, 2, 25, 3))
    for p in range(2):
        for j in range(25):
            for c in range(3):
                f = interp1d(x_old, sequence[:, p, j, c], kind="linear")
                out[:, p, j, c] = f(x_new)
    return out


def select_active_frames(sequence, target_frames):
    if len(sequence) <= target_frames:
        return temporal_interpolation(sequence, target_frames)

    motion = np.sum(
        np.abs(np.diff(sequence[:, :, :, :2], axis=0)), axis=(1, 2, 3)
    )
    idx = np.argsort(motion)[-target_frames:]
    return sequence[idx]


def process_video(input_dir, output_dir, target_frames=10):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_frame(f) for f in sorted(input_dir.glob("*.json"))]
    sequence = np.array(frames)

    sequence = interpolate_missing(sequence)
    sequence = select_active_frames(sequence, target_frames)

    for i, frame in enumerate(sequence):
        out = {"version": 1.3, "people": []}
        for p in range(2):
            if np.any(frame[p]):
                out["people"].append({
                    "pose_keypoints_2d": frame[p].flatten().tolist()
                })

        with open(output_dir / f"frame_{i:04d}.json", "w") as f:
            json.dump(out, f)
