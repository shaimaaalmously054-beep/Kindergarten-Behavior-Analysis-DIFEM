import numpy as np

joint_indices = [
    4, 7, 3, 6, 8, 11, 9, 12, 10, 13, 1  # نفس المفاصل المختارة
]

def calculate_joint_angles(frames):
    angles = []
    for frame in frames:
        person1 = frame[0]
        angle_sum = 0
        valid_angles = 0
        for j1_idx, j2_idx, j3_idx in [(2, 3, 4), (5, 6, 7)]:
            x1, y1 = person1[j1_idx, :2]
            x2, y2 = person1[j2_idx, :2]
            x3, y3 = person1[j3_idx, :2]
            if person1[j1_idx, 2] > 0 and person1[j2_idx, 2] > 0 and person1[j3_idx, 2] > 0:
                v1 = np.array([x2 - x1, y2 - y1])
                v2 = np.array([x3 - x2, y3 - y2])
                dot_product = np.dot(v1, v2)
                mag_v1 = np.sqrt(np.sum(v1**2))
                mag_v2 = np.sqrt(np.sum(v2**2))
                cos_angle = dot_product / (mag_v1 * mag_v2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                angle_sum += angle
                valid_angles += 1
        angles.append(angle_sum / valid_angles if valid_angles > 0 else 0)
    return np.array(angles)

def calculate_joint_overlap(frames):
    overlaps = []
    joint_indices = list(range(25))  # لجميع المفاصل
    for frame in frames:
        person1, person2 = frame[0], frame[1]
        if np.any(person2[:, 2] == 0):
            overlaps.append(0)
            continue
        valid_joints = person2[person2[:, 2] > 0, :2]
        if len(valid_joints) == 0:
            overlaps.append(0)
            continue
        x_min, y_min = np.min(valid_joints, axis=0)
        x_max, y_max = np.max(valid_joints, axis=0)
        overlap_count = 0
        for j_idx in joint_indices:
            x, y = person1[j_idx, :2]
            if person1[j_idx, 2] > 0:
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    overlap_count += 1
        overlaps.append(overlap_count)
    return np.array(overlaps)
