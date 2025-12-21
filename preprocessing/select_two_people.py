"""
Selects the most relevant two interacting persons per frame
based on motion, proximity, overlap, and confidence.
This step ensures a fixed number of persons for DIFEM.
"""

import os
import json
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ---------------- Utility functions ---------------- #

def calculate_center_of_mass(keypoints, conf_th=0.3):
    keypoints = np.array(keypoints).reshape(-1, 3)
    valid = keypoints[keypoints[:, 2] > conf_th]
    if len(valid) == 0:
        return None
    return np.mean(valid[:, :2], axis=0)


def calculate_bounding_box(keypoints, conf_th=0.3):
    keypoints = np.array(keypoints).reshape(-1, 3)
    valid = keypoints[keypoints[:, 2] > conf_th]
    if len(valid) == 0:
        return None
    x_min, y_min = np.min(valid[:, :2], axis=0)
    x_max, y_max = np.max(valid[:, :2], axis=0)
    return (x_min, y_min, x_max, y_max)


def check_overlap(box1, box2):
    if box1 is None or box2 is None:
        return False
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return (
        x1_min <= x2_max and x1_max >= x2_min and
        y1_min <= y2_max and y1_max >= y2_min
    )


def calculate_hand_to_body_proximity(kp1, kp2, conf_th=0.3):
    kp1 = np.array(kp1).reshape(-1, 3)
    kp2 = np.array(kp2).reshape(-1, 3)

    hand_ids = [4, 7]
    torso_ids = [1, 8]

    min_dist = np.inf
    for h in hand_ids:
        if kp1[h, 2] <= conf_th:
            continue
        for t in torso_ids:
            if kp2[t, 2] <= conf_th:
                continue
            dist = np.linalg.norm(kp1[h, :2] - kp2[t, :2])
            min_dist = min(min_dist, dist)

    return min_dist if min_dist != np.inf else 1e6


def calculate_motion_speed(curr_kp, prev_kp, conf_th=0.3):
    if prev_kp is None:
        return 0.0

    curr = np.array(curr_kp).reshape(-1, 3)
    prev = np.array(prev_kp).reshape(-1, 3)

    motion_joints = [4, 7, 11, 14]
    motion = []

    for j in motion_joints:
        if curr[j, 2] > conf_th and prev[j, 2] > conf_th:
            motion.append(np.linalg.norm(curr[j, :2] - prev[j, :2]))

    return np.mean(motion) if motion else 0.0


def match_previous_person(center, prev_people):
    if center is None or not prev_people:
        return None

    best_dist = np.inf
    best_person = None

    for p in prev_people:
        kp = p.get("pose_keypoints_2d", [])
        if not kp:
            continue
        prev_center = calculate_center_of_mass(kp)
        if prev_center is None:
            continue
        d = np.linalg.norm(center - prev_center)
        if d < best_dist:
            best_dist = d
            best_person = p

    return best_person if best_dist < 0.1 else None


# ---------------- Core logic ---------------- #

def select_two_people(json_data, prev_json=None):
    people = json_data.get("people", [])
    if len(people) <= 2:
        return json_data

    centers, boxes, motions, confidences = [], [], [], []
    valid_people = []

    for p in people:
        kp = p.get("pose_keypoints_2d", [])
        if not kp:
            continue

        center = calculate_center_of_mass(kp)
        if center is None:
            continue

        box = calculate_bounding_box(kp)
        conf = np.mean(np.array(kp)[2::3])

        prev_p = match_previous_person(center, prev_json["people"]) if prev_json else None
        prev_kp = prev_p.get("pose_keypoints_2d") if prev_p else None
        motion = calculate_motion_speed(kp, prev_kp)

        centers.append(center)
        boxes.append(box)
        motions.append(motion)
        confidences.append(conf)
        valid_people.append(p)

    if len(valid_people) < 2:
        json_data["people"] = valid_people
        return json_data

    scores = []
    for i in range(len(valid_people)):
        for j in range(i + 1, len(valid_people)):
            dist = np.linalg.norm(centers[i] - centers[j])
            overlap = check_overlap(boxes[i], boxes[j])
            prox = min(
                calculate_hand_to_body_proximity(
                    valid_people[i]["pose_keypoints_2d"],
                    valid_people[j]["pose_keypoints_2d"]
                ),
                calculate_hand_to_body_proximity(
                    valid_people[j]["pose_keypoints_2d"],
                    valid_people[i]["pose_keypoints_2d"]
                )
            )

            score = (
                (1 / (dist + 1e-6)) * 20 +
                (10 if overlap else 0) +
                (1 / (prox + 1e-6)) * 15 +
                (motions[i] + motions[j]) * 5
            )
            scores.append((score, i, j))

    _, i, j = max(scores, key=lambda x: x[0])
    json_data["people"] = [valid_people[i], valid_people[j]]
    return json_data


def process_json_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))

    prev_data = None
    for f in files:
        with open(os.path.join(input_dir, f)) as jf:
            data = json.load(jf)

        data = select_two_people(data, prev_data)
        prev_data = data

        with open(os.path.join(output_dir, f), "w") as out:
            json.dump(data, out)

    logging.info("Finished selecting two people per frame.")
