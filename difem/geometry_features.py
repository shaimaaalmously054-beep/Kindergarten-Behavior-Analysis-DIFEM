"""
Geometry-based features for DIFEM
Joint angles and overlap computation.
"""

import numpy as np


def calculate_joint_angles(frames):
    angles = []

    for frame in frames:
        person = frame[0]
        angle_sum = 0
        valid_angles = 0

        for j1, j2, j3 in [(2, 3, 4), (5, 6, 7)]:  # shoulders-elbows-wrists
            if person[j1, 2] > 0 and person[j2, 2] > 0 and person[j3, 2] > 0:
                v1 = person[j2, :2] - person[j1, :2]
                v2 = person[j3, :2] - person[j2, :2]

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

                angle_sum += angle
                valid_angles += 1

        angles.append(angle_sum / valid_angles if valid_angles > 0 else 0)

    return np.array(angles)


def calculate_joint_overlap(frames, joint_indices):
    overlaps = []

    for frame in frames:
        person1, person2 = frame[0], frame[1]

        if np.any(person2[:, 2] == 0):
            overlaps.append(0)
            continue

        valid_joints = person2[person2[:, 2] > 0, :2]
        x_min, y_min = valid_joints.min(axis=0)
        x_max, y_max = valid_joints.max(axis=0)

        overlap_count = 0
        for j_idx in joint_indices:
            if person1[j_idx, 2] > 0:
                x, y = person1[j_idx, :2]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    overlap_count += 1

        overlaps.append(overlap_count)

    return np.array(overlaps)
