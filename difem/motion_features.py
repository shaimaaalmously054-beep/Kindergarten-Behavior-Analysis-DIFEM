"""
Motion-based features for DIFEM
"""

import numpy as np


def compute_velocity(skeleton_sequence):
    velocities = []
    for t in range(1, len(skeleton_sequence)):
        velocities.append(
            np.linalg.norm(
                np.array(skeleton_sequence[t]) -
                np.array(skeleton_sequence[t - 1])
            )
        )
    return velocities


def compute_acceleration(velocities):
    accelerations = []
    for t in range(1, len(velocities)):
        accelerations.append(velocities[t] - velocities[t - 1])
    return accelerations
