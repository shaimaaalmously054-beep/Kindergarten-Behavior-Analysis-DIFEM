"""
Dynamic Image Feature Extraction Method (DIFEM)

Author: Shaimaa Almously
Graduation Project – 2025
"""

import os
import json
import numpy as np

from difem.motion_features import calculate_velocity, calculate_acceleration
from difem.geometry_features import calculate_joint_angles, calculate_joint_overlap


SELECTED_JOINTS = {
    'right_wrist': 4, 'left_wrist': 7, 'right_elbow': 3, 'left_elbow': 6,
    'right_hip': 8, 'left_hip': 11, 'right_knee': 9, 'left_knee': 12,
    'right_ankle': 10, 'left_ankle': 13, 'neck': 1
}

JOINT_WEIGHTS = {
    'right_wrist': 1.0, 'left_wrist': 1.0,
    'right_elbow': 0.8, 'left_elbow': 0.8,
    'right_hip': 1.0, 'left_hip': 1.0,
    'right_knee': 1.0, 'left_knee': 1.0,
    'right_ankle': 1.0, 'left_ankle': 1.0,
    'neck': 1.0
}

JOINT_INDICES = list(SELECTED_JOINTS.values())
JOINT_WEIGHT_LIST = list(JOINT_WEIGHTS.values())


def load_keypoints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data["people"]:
        return np.zeros((2, 25, 3))

    persons = []
    for i in range(min(2, len(data["people"]))):
        persons.append(
            np.array(data["people"][i]["pose_keypoints_2d"]).reshape(25, 3)
        )

    while len(persons) < 2:
        persons.append(np.zeros((25, 3)))

    return np.stack(persons)


def extract_difem_features(json_folder):
    frame_files = sorted(f for f in os.listdir(json_folder) if f.endswith(".json"))
    frames = np.array([load_keypoints(os.path.join(json_folder, f)) for f in frame_files])

    velocities = calculate_velocity(frames, JOINT_INDICES, JOINT_WEIGHT_LIST)
    accelerations = calculate_acceleration(velocities)
    overlaps = calculate_joint_overlap(frames, JOINT_INDICES)
    angles = calculate_joint_angles(frames)

    return np.array([
        np.mean(velocities), np.max(velocities), np.var(velocities),
        np.mean(overlaps), np.var(overlaps),
        np.mean(accelerations), np.max(accelerations), np.var(accelerations),
        np.mean(angles), np.var(angles)
    ])
