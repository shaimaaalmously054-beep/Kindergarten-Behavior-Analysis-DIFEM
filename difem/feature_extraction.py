"""
Dynamic Image Feature Extraction Method (DIFEM)
-----------------------------------------------
Skeleton-based feature extraction for behavior analysis.

Author: Shaima Almously
Graduation Project – 2025

Note:
This file contains only my individual contribution and does not
include behavioral decision logic implemented by other team members.
"""

import os
import numpy as np
import json
from difem.motion_features import calculate_velocity, calculate_acceleration
from difem.geometry_features import calculate_joint_angles, calculate_joint_overlap

joint_indices = list(range(25))  # لجميع المفاصل

def load_keypoints(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if data['people']:
            if len(data['people']) >= 2:
                person1 = np.array(data['people'][0]['pose_keypoints_2d']).reshape(25, 3)
                person2 = np.array(data['people'][1]['pose_keypoints_2d']).reshape(25, 3)
                return np.stack([person1, person2])
            else:
                person1 = np.array(data['people'][0]['pose_keypoints_2d']).reshape(25, 3)
                return np.stack([person1, np.zeros((25, 3))])
        return np.zeros((2, 25, 3))
    except Exception as e:
        print(f"خطأ في تحميل {file_path}: {e}")
        return np.zeros((2, 25, 3))

def extract_difem_features(folder_path):
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    if len(frame_files) != 10:
        print(f"تخطي {folder_path}: وجد {len(frame_files)} فريمات بدلاً من 10")
        return None

    frames = []
    for file in frame_files:
        frame_data = load_keypoints(os.path.join(folder_path, file))
        if np.any(frame_data):
            frames.append(frame_data)
        else:
            print(f"بيانات فريم غير صالحة في {file}")
            return None

    frames = np.array(frames)
    if len(frames) != 10:
        print(f"تخطي {folder_path}")
        return None

    velocities = calculate_velocity(frames)
    accelerations = calculate_acceleration(velocities)
    overlaps = calculate_joint_overlap(frames)
    angles = calculate_joint_angles(frames)

    return np.array([
        np.mean(velocities), np.max(velocities), np.var(velocities),
        np.mean(overlaps), np.var(overlaps),
        np.mean(accelerations), np.max(accelerations), np.var(accelerations),
        np.mean(angles), np.var(angles)
    ])
