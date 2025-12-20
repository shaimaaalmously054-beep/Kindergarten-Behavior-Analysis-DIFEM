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

from difem.motion_features import compute_velocity, compute_acceleration
from difem.geometry_features import compute_joint_angles, compute_overlap


class DIFEM:
    def __init__(self, num_joints=12):
        self.num_joints = num_joints

    def extract(self, skeleton_sequence):
        """
        Extract DIFEM features from a sequence of skeleton frames.

        Args:
            skeleton_sequence (list): List of skeleton frames

        Returns:
            dict: Extracted DIFEM features
        """
        features = {}

        features["velocity"] = compute_velocity(skeleton_sequence)
        features["acceleration"] = compute_acceleration(skeleton_sequence)
        features["angles"] = compute_joint_angles(skeleton_sequence)
        features["overlap"] = compute_overlap(skeleton_sequence)

        return features
