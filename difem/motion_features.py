"""
Motion-based features for DIFEM
Velocity and acceleration computation from skeleton sequences.
"""

import numpy as np


def calculate_velocity(frames, joint_indices, joint_weights):
    velocities = []

    for i in range(len(frames) - 1):
        frame_t = frames[i]
        frame_t1 = frames[i + 1]
        frame_velocities = []

        for p in range(frame_t.shape[0]):
            person_velocities = []
            for j_idx, weight in zip(joint_indices, joint_weights):
                x_t, y_t = frame_t[p, j_idx, :2]
                x_t1, y_t1 = frame_t1[p, j_idx, :2]

                if frame_t[p, j_idx, 2] > 0 and frame_t1[p, j_idx, 2] > 0:
                    velocity = np.sqrt(weight * ((x_t1 - x_t) ** 2 + (y_t1 - y_t) ** 2))
                else:
                    velocity = 0

                person_velocities.append(velocity)

            frame_velocities.append(np.mean(person_velocities))

        velocities.append(np.mean(frame_velocities))

    return np.array(velocities)


def calculate_acceleration(velocities):
    return np.diff(velocities)
