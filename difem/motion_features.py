import numpy as np

# المفاصل المختارة وأوزانها
selected_joints = {
    'right_wrist': 4, 'left_wrist': 7, 'right_elbow': 3, 'left_elbow': 6,
    'right_hip': 8, 'left_hip': 11, 'right_knee': 9, 'left_knee': 12,
    'right_ankle': 10, 'left_ankle': 13, 'neck': 1
}

joint_weights = {
    'right_wrist': 1.0, 'left_wrist': 1.0, 'right_elbow': 0.8, 'left_elbow': 0.8,
    'right_hip': 1.0, 'left_hip': 1.0, 'right_knee': 1.0, 'left_knee': 1.0,
    'right_ankle': 1.0, 'left_ankle': 1.0, 'neck': 1.0
}

joint_indices = list(selected_joints.values())

def calculate_velocity(frames):
    velocities = []
    for i in range(len(frames)-1):
        frame_t = frames[i]
        frame_t1 = frames[i+1]
        frame_velocities = []
        for p in range(frame_t.shape[0]):
            person_velocities = []
            for j_idx in joint_indices:
                x_t, y_t = frame_t[p, j_idx, :2]
                x_t1, y_t1 = frame_t1[p, j_idx, :2]
                weight = joint_weights[list(selected_joints.keys())[joint_indices.index(j_idx)]]
                if frame_t[p, j_idx, 2] > 0 and frame_t1[p, j_idx, 2] > 0:
                    velocity = np.sqrt(weight * ((x_t1 - x_t)**2 + (y_t1 - y_t)**2))
                else:
                    velocity = 0
                person_velocities.append(velocity)
            frame_velocities.append(np.mean(person_velocities))
        velocities.append(np.mean(frame_velocities))
    return np.array(velocities)

def calculate_acceleration(velocities):
    accelerations = []
    for i in range(len(velocities)-1):
        accel = velocities[i+1] - velocities[i]
        accelerations.append(accel)
    return np.array(accelerations)
