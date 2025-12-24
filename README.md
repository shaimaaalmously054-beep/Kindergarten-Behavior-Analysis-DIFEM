# Kindergarten-Behavior-Analysis-DIFEM

An intelligent system for analyzing children’s behavior in kindergarten
using computer vision and machine learning.
The project implements the **Dynamic Image Feature Extraction Method (DIFEM)**
to extract motion and geometry features from 12 skeletal joints using OpenPose,
and applies a Random Forest classifier achieving **93.6% accuracy**.

---

## 📌 Overview
This repository presents the **Dynamic Image Feature Extraction Method (DIFEM)**,
developed as part of a graduation project (2025), for analyzing children’s behavior
in kindergarten environments using **skeleton-based computer vision features**.

The system extracts meaningful **motion and geometry descriptors** from OpenPose
keypoint sequences to support behavior recognition tasks.

---

## 🧠 DIFEM – Dynamic Image Feature Extraction Method
DIFEM is a handcrafted feature extraction method designed to capture:

- Velocity and acceleration of selected body joints
- Joint angle dynamics
- Physical overlap between interacting children
- Temporal motion patterns across frames

The method uses **12 key skeletal joints** and produces a compact numerical
representation suitable for classical machine learning models.

---

⚙️ Processing Pipeline
Frame Extraction
Extract frames from the video at a specified interval (e.g., every 1 second).

Skeleton Extraction
Use OpenPose to extract 2D skeleton keypoints for each extracted video frame.

Preprocessing

Selection of the two most relevant interacting persons per frame.
Temporal normalization of skeleton sequences to a fixed length (default: 10 frames).
Feature Extraction (DIFEM)

Motion-based features (velocity, acceleration).
Geometry-based features (joint angles, overlap).
Behavior Classification

Random Forest classifier.
---

## 📂 Project Structure

