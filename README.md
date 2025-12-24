# Kindergarten-Behavior-Analysis-DIFEM

An intelligent system for analyzing children’s behavior in kindergarten using computer vision and machine learning.  
The project implements the **Dynamic Image Feature Extraction Method (DIFEM)** to extract motion and geometry features from skeletal data, and validates the extracted features using a Random Forest classifier achieving **up to 93.6% accuracy**.

---

## 📌 Overview
This repository presents the **Dynamic Image Feature Extraction Method (DIFEM)**, developed as part of a graduation project (2025), for analyzing children’s behavior in kindergarten environments using **skeleton-based computer vision features**.

The focus of this repository is on **feature extraction and preprocessing**, rather than end-to-end video processing.

---

## 🧠 DIFEM – Dynamic Image Feature Extraction Method
DIFEM is a handcrafted feature extraction method designed to capture:

- Velocity and acceleration of selected body joints  
- Joint angle dynamics  
- Physical overlap between interacting children  
- Temporal motion patterns across frames  

The method uses **12 selected skeletal joints** and produces a compact numerical feature vector suitable for classical machine learning models.

---

## ⚙️ Processing Pipeline

> ⚠️ **Note:** Frame extraction and OpenPose execution are assumed as pre-processing.

1. **Frame Extraction**  
   Video frames should be extracted at a fixed interval (e.g., 1 frame per second).

2. **Skeleton Extraction**  
   OpenPose is used to extract 2D skeleton keypoints for each frame.  
   The output should be a sequence of OpenPose JSON files.

3. **Preprocessing**  
   - Selection of the two most relevant interacting persons per frame  
   - Temporal normalization of skeleton sequences to a fixed length (default: 10 frames)

4. **Feature Extraction (DIFEM)**  
   - Motion-based features (velocity, acceleration)  
   - Geometry-based features (joint angles, overlap)

5. **Behavior Classification (Demo / Validation)**  
   - Random Forest classifier used to validate DIFEM features  
   - The classifier is provided for demonstration purposes only

---

## 📂 Project Structure

```text
Kindergarten-Behavior-Analysis-DIFEM/
│
├── difem/                      # DIFEM core implementation
│   ├── feature_extraction.py
│   ├── motion_features.py
│   ├── geometry_features.py
│   └── utils.py
│
├── preprocessing/              # Skeleton preprocessing
|   ├── frame_extraction.py
|   ├── openpose_runner.py
│   ├── select_two_people.py
│   └── frame_normalization.py
│
├── models/                     # Demo models (optional)
│   ├── random_forest_difem.pkl
|   ├── feature_means_violence.npy
|   ├── feature_means_nonviolence.npy
│   └── scaler.pkl
│
├── example_usage.py            # End-to-end DIFEM usage example
├── requirements.txt
├── README.md
├── DEMO.mp4
└── LICENSE
