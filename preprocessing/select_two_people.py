"""
Selects the two most relevant interacting persons per frame.
Criteria: proximity, overlap, motion, and confidence.
Ensures fixed input (2 persons) for DIFEM.
"""

import os
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CONF_TH = 0.3
HAND_IDS = [4, 7]
TORSO_IDS = [1, 8]
MOTION_JOINTS = [4, 7, 11, 14]


# ---------------- Utility functions ---------------- #

def center_of_mass(keypoints):
    kp = np.array(keypoints).reshape(-1, 3)
    valid = kp[kp[:, 2] > CONF_TH]
    return np.mean(valid[:, :2], axis=0) if len(valid) else None


def bounding_box(keypoints):
    kp = np.array(keypoints).reshape(-1, 3)
    valid = kp[kp[:, 2] > CONF_TH]
    if len(valid) == 0:
        return None
    x_min, y_min = valid[:, :2].min(axis=0)
    x_max, y_max = valid[:, :2].max(axis=0)
    return x_min, y_min, x_max, y_max


def overlap(box1, box2):
    if box1 is None or box2 is None:
        return False
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return x1 <= x4 and x2 >= x3 and y1 <= y4 and y2 >= y3


def hand_to_body_distance(kp1, kp2):
    kp1 = np.array(kp1).reshape(-1, 3)
    kp2 = np.array(kp2).reshape(-1, 3)

    dists = []
    for h in HAND_IDS:
        if kp1[h, 2] <= CONF_TH:
            continue
        for t in TORSO_IDS:
            if kp2[t, 2] <= CONF_TH:
                continue
            dists.append(np.linalg.norm(kp1[h, :2] - kp2[t, :2]))

    return min(dists) if dists else 1e6


def motion_speed(curr, prev):
    if prev is None:
        return 0.0

    curr = np.array(curr).reshape(-1, 3)
    prev = np.array(prev).reshape(-1, 3)

    speeds = []
    for j in MOTION_JOINTS:
        if curr[j, 2] > CONF_TH and prev[j, 2] > CONF_TH:
            speeds.append(np.linalg.norm(curr[j, :2] - prev[j, :2]))

    return np.mean(speeds) if speeds else 0.0


# ---------------- Core logic ---------------- #

def select_two_people(json_data, prev_data=None):
    people = json_data.get("people", [])
    if len(people) <= 2:
        return json_data

    centers, boxes, motions, valid = [], [], [], []

    for p in people:
        kp = p.get("pose_keypoints_2d", [])
        center = center_of_mass(kp)
        if center is None:
            continue

        box = bounding_box(kp)
        prev_kp = None
        if prev_data:
            for prev_p in prev_data.get("people", []):
                prev_kp = prev_p.get("pose_keypoints_2d", None)

        centers.append(center)
        boxes.append(box)
        motions.append(motion_speed(kp, prev_kp))
        valid.append(p)

    if len(valid) < 2:
        json_data["people"] = valid
        return json_data

    scores = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            dist = np.linalg.norm(centers[i] - centers[j])
            score = (
                (1 / (dist + 1e-6)) * 20 +
                (10 if overlap(boxes[i], boxes[j]) else 0) +
                (1 / (hand_to_body_distance(
                    valid[i]["pose_keypoints_2d"],
                    valid[j]["pose_keypoints_2d"]
                ) + 1e-6)) * 15 +
                (motions[i] + motions[j]) * 5
            )
            scores.append((score, i, j))

    _, i, j = max(scores, key=lambda x: x[0])
    json_data["people"] = [valid[i], valid[j]]
    return json_data


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))

    prev = None
    for f in files:
        with open(os.path.join(input_dir, f)) as jf:
            data = json.load(jf)

        data = select_two_people(data, prev)
        prev = data

        with open(os.path.join(output_dir, f), "w") as out:
            json.dump(data, out)

    logging.info("Two-person selection completed.")
