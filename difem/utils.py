import json
import numpy as np


def load_keypoints(json_path, max_people=2):
    with open(json_path) as f:
        data = json.load(f)

    frames = np.zeros((max_people, 25, 3))
    people = data.get("people", [])

    for i in range(min(max_people, len(people))):
        kp = np.array(people[i]["pose_keypoints_2d"]).reshape(25, 3)
        frames[i] = kp

    return frames


def load_video_sequence(json_dir):
    import os
    files = sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))
    sequence = [load_keypoints(os.path.join(json_dir, f)) for f in files]
    return np.array(sequence)
